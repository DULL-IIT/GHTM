import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import NMF
import argparse
from torch_geometric.data import Data
import pickle
import torch.nn as nn
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import dropout_edge
import torch.nn.functional as F
from torch_geometric.nn import ClusterGCNConv
import random

import os
from torch_geometric.loader import ClusterData, ClusterLoader

import requests
import zipfile
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def bn_glove_model():

    os.makedirs('model', exist_ok=True)
    
    
    model_url = "https://huggingface.co/sagorsarker/bangla-glove-vectors/resolve/main/bn_glove.39M.300d.zip"
    zip_path = "model/bn_glove.39M.300d.zip"
    extracted_path = "model/bn_glove.39M.300d.txt"
    
    if os.path.exists(extracted_path):
        print("Model file already exists. Skipping download.")
        return
    
    print("Downloading GloVe model...")

    response = requests.get(model_url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed.")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")
    
    print("Extracting model...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('model/')
    print("Extraction completed.")
    

    os.remove(zip_path)
    print("Clean up completed.")

def bengali_tokenizer(text):
    return text.split()

def build_knn_graph(embeddings: np.ndarray, k: int):
    """
    embeddings: [num_nodes, 300] numpy array of text embeddings
    k: number of neighbors for k-NN
    returns: edge_index tensor for PyG
    """
    A = kneighbors_graph(embeddings, n_neighbors=k, mode='distance', metric='cosine', n_jobs=-1, include_self=True)      

    A = A.tocoo()
    edge_index = torch.tensor([A.row, A.col], dtype=torch.long)
    x = torch.tensor(embeddings, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index)

class ClusterGCNNet(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.4,
        edge_dropout: float = 0.2,      
    ):
        super().__init__()
        self.input_drop = nn.Dropout(p=dropout)            
    
        self.convs = nn.ModuleList([
            ClusterGCNConv(in_dim, hidden_dim),
            ClusterGCNConv(hidden_dim, hidden_dim),
            ClusterGCNConv(hidden_dim, hidden_dim),
            ClusterGCNConv(hidden_dim, out_dim),
        ])
    
        self.norms = nn.ModuleList([
            GraphNorm(hidden_dim),
            GraphNorm(hidden_dim),
            GraphNorm(hidden_dim),
            GraphNorm(out_dim),
        ])                                                
        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def forward(self, x, edge_index):

        for i, conv in enumerate(self.convs):
        
            edge_index, _ = dropout_edge(
                edge_index, p=self.edge_dropout, training=self.training
            )                                             
 
            h = conv(x, edge_index)

            h = self.norms[i](h)

            if i != len(self.convs) - 1:          
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            if h.shape == x.shape:
                h = h + x                                            

            x = h

        return x  
    
def get_cluster_loader(data, num_clusters, batch_size):
    """
    data: PyG Data object
    num_clusters: into how many partitions to split the full graph
    batch_size: how many clusters per mini-batch
    """
    
    cluster_data = ClusterData(data, num_parts=num_clusters, keep_inter_cluster_edges=True)   
    loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
    
    return loader

def hinge_edge_loss(z, edge_index, num_neg_samples=5, margin=1.0):
    """
    z: [N, out_dim] node embeddings
    edge_index: [2, E] positive edges in this mini-batch
    For each positive edge (i,j), sample `num_neg_samples` negative j'.
    """
    
    src, dst = edge_index     

    pos_score = F.cosine_similarity(z[src], z[dst], dim=1)

    E = dst.size(0)
    neg_dst = dst[torch.randint(0, E, (E * num_neg_samples,))]

    neg_src = src.repeat_interleave(num_neg_samples)   

    neg_score = F.cosine_similarity(z[neg_src], z[neg_dst], dim=1)

    loss = torch.relu(margin - pos_score.repeat(num_neg_samples) + neg_score)

    return loss.mean()


def contrastive_loss(z, temperature=0.5):
    """
    z: [N_b, d] embeddings for the batch
    Returns NT-Xent loss that pulls each embedding towards itself under 
    two random augmentations (here we simply treat each node once).
    """

    z_norm = F.normalize(z, p=2, dim=1)                         

    sim = torch.matmul(z_norm, z_norm.T) / temperature         

    mask = torch.eye(sim.size(0), device=sim.device).bool()
    sim.masked_fill_(mask, -9e15)

    labels = torch.arange(sim.size(0), device=sim.device)

    return F.cross_entropy(sim, labels)                        


def joint_loss(z, edge_index):
  
    loss_hinge = hinge_edge_loss(z, edge_index)

    loss_con = contrastive_loss(z)

    return loss_hinge + loss_con

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z = model(batch.x, batch.edge_index)
        loss = joint_loss(z,batch.edge_index)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
    return total_loss / len(loader.dataset)

def extract_embeddings(model, data):
    model.eval()                                   
    data = data.to(device)
    with torch.no_grad():                         
        z = model(data.x, data.edge_index)  
        
    return z.cpu().numpy()

def main(args):
    ## CONFIG ##
    num_topics = args.num_topics
    top_n_words = args.top_n_words
    rep_words = args.rep_words
    
    hidden_dim = args.hidden_dim
    output_dim = 64

    n_neighbors = args.n_neighbors
    num_clusters = args.num_clusters

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size


    with open('./data/processed_data.pickle', 'rb') as f:
        docs = pickle.load(f)
    random.shuffle(docs)

    print("Data Loaded.")

    ## Text Vectorization ##

    print("Text Vectorization Stage....")
    bn_glove_model()

    glove = {}
    with open('./model/bn_glove.39M.300d.txt', encoding='utf8') as f:
        for L in f:
            parts = L.rstrip().split()
            glove[parts[0]] = np.array(parts[1:], dtype=float)
    emb_dim = len(next(iter(glove.values())))

    vectorizer = TfidfVectorizer(
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        tokenizer=bengali_tokenizer,
        ngram_range=(1,1)           ### Increase N-Gram range for better results
    )

    docs_as_strings = [' '.join(doc) for doc in docs]

    sparse = vectorizer.fit_transform(docs_as_strings)
    terms = vectorizer.get_feature_names_out()

    G = np.array([glove.get(t, np.zeros(emb_dim)) for t in terms])
    doc_embeds = sparse.dot(G)

    ## Graph-based Representation Learning

    print("Graph-based Representation Learning Stage....")

    data = build_knn_graph(doc_embeds, k=n_neighbors)

    loader = get_cluster_loader(data, num_clusters=num_clusters, batch_size=batch_size)
    model = ClusterGCNNet(doc_embeds.shape[1], hidden_dim=hidden_dim, out_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = train(model, loader, optimizer, device)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.2f}")

    reduced = extract_embeddings(model, data)


    ## Matrix Factorization ##

    print("Matrix Factorization Stage....")

    non_neg_embeddings = np.abs(reduced)

    topic_words = []
    nmf = NMF(n_components=num_topics, random_state=42, init='nndsvda', max_iter=500)
    doc_topic = nmf.fit_transform(non_neg_embeddings)       

    ## Topic Representation ##

    print("Topic Representation Stage....")
    
    for t in range(num_topics):
        best_docs = np.argsort(doc_topic[:, t])[::-1][:rep_words]
        
        topic_term_weights = np.asarray(sparse[best_docs].sum(axis=0)).ravel()
        
        top_terms = terms[np.argsort(topic_term_weights)[::-1][:top_n_words]]
        topic_words.append(top_terms.tolist())

        print(f"Topic {t:>2}:", top_terms.tolist())



if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_topics', type=int, default=8)
    parser.add_argument('--top_n_words', type=int, default=10)
    parser.add_argument('--rep_words', type=int, default=20)

    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--n_neighbors', type=int, default=15)
    parser.add_argument('--num_clusters', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=float, default=32)

    args = parser.parse_args()

    main(args)
