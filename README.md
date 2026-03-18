# GHTM-Graph-based-Hybrid-Topic-Modeling

This repo contains the work behind the paper:  
**GHTM: A graph-based hybrid topic modeling approach with a benchmark dataset for the low-resource Bengali language**


The GHTM model consists of three stages:

- **Text Vectorization Stage**: Converts the text documents into a sum of TF-IDF weighted GloVe embeddings. 

- **Graph-based Representation Learning Stage**: Generates a KNN graph and separates it into sub-graphs. Document vectors, represented as graph nodes, are enriched by the GCN layers.

- **Non-negative Matrix Factorization Stage**: Accomplishes topic modeling by first performing an absolute value transformation on the newly generated embedding matrix and then factorizing it into two matrices. 

- **Topic Representation Stage**: Identifies the top representative documents and chooses a collection of keywords to represent the topics.
---

## Installation

```bash
git clone https://github.com/DULL-IIT/GHTM/
cd GHTM
```
If you prefer conda, then run:
```
conda env create -f environment.yml
conda activate ghtm
python setup.py
```

Or you can try pip:

```
pip install -r requirements.txt
python setup.py
```

## Prepare Data

Run the following commands to prepare Bengali data for GHTM:

```bash
cd GHTM
python data_prep.py --file_path ../datasets/NCTBText/NCTBText.csv --text_column text

```
After this step, clean and tokenized data will be saved in `GHTM/data` as `processed_data.pickle`, prepared for action.

## GloVe Model Placement

- GloVe model for Bengali will be automatically downloaded and placed in `GHTM/model/`

To manually do it for other languages:

- Download the GloVe model

- Place the model file inside the directory: `GHTM/model/`

Make sure the path matches what’s expected in the notebook.


## Train the Model

To train GHTM and generate topic words, run:

```bash
python ghtm.py --num_topics 8 --top_n_words 10 --num_clusters 4 --epochs 100
```

Note: 
- Notebooks are available in `GHTM/notebooks` for demonstration.
- `environment.yml` is written to set up the conda environment for running GHTM only. The experiments done for the comprehensive benchmarking are demonstrated in `experiments`, but the environment is not set up for running the experiments.


---
# NCTBText: Bengali Dataset

This dataset contains Bengali textual data curated from NCTB (National Curriculum and Textbook Board) textbooks, which are widely used in Bangladeshi schools. 
Source materials are derived from: https://nctb.gov.bd/ 

This dataset was curated as a benchmark for Bengali topic modeling and to support research in other Bengali NLP tasks.

The raw text was extracted from official NCTB textbooks and processed using multiple stages of cleaning and filtering.

- **File**: `NCTBText.csv`
- **Location**: `../datasets/NCTBText/NCTBText.csv`
- **Columns**:
  - `text`: Cleaned paragraph-level text data extracted from Bengali textbooks.
  - `label`: Subject/Topic of the text.
---
