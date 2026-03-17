# GHTM-Graph-based-Hybrid-Topic-Modeling

This repo contains the work behind the paper:  
**GHTM: A Graph-based Hybrid Topic Modeling Approach in Low-Resource Bengali Language**

---

## Overview

GHTM model consists of three stages:

- **Text Feature Extraction**: using a combination of TF-IDF and GloVe representations.
- **Graph Construction & GCN**: a graph is built from the extracted vectors, which is processed by a Graph Convolutional Network (GCN) to produce similarity-aware embeddings.
- **Matrix Factorization**: applied to the embeddings to generate diverse and coherent topic keywords.

This hybrid approach helps GHTM uncover coherent, structure-aware latent topics.

---

## Installation

```bash
git clone https://github.com/kya602/GHTM-Graph-based-Hybrid-Topic-Modeling.git
cd GHTM-Graph-based-Hybrid-Topic-Modeling

pip install -r requirements.txt
```

---

## Prepare Data

Run the following notebook to prepare data:

```bash
jupyter notebook GHTM/Data Preparation.ipynb
```

This is a reference notebook for preprocessing and preparing data.

---

## Models and Files Placement

- Download the GloVe models for Bangla from:  
  🔗 https://github.com/sagorbrur/GloVe-Bengali

- Place the model file inside the following directory:

```
GHTM/model/
```

Make sure the path matches what’s expected in the notebook.

---

## Train the Model

To train the topic model and generate topic words, run:

```bash
jupyter notebook GHTM/GHTM.ipynb
```

This notebook shows:
- How to train the model for topic modeling.
- How to evaluate the model and extract topic keywords.

---
# NCTBText: Bengali Dataset

This dataset contains Bengali textual data curated from NCTB (National Curriculum and Textbook Board) textbooks, which are widely used in Bangladeshi schools. 
Source materials are derived from: https://nctb.gov.bd/ 

---

## Overview

This dataset was curated to support research in Bengali NLP tasks such as:
- Topic Modeling
- Text Classification

The raw text was extracted from official NCTB textbooks and processed using multiple stages of cleaning and filtering.

- **File**: `NCTBText.csv`
- **Location**: `../datasets/NCTBText/NCTBText.csv`
- **Columns**:
  - `text`: Cleaned paragraph-level text data extracted from Bengali textbooks.
  - `label`: Subject/Topic of the text.
---
