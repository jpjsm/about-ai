# about-ai

A place to save knowledge about AI, transformers, RAG, LLM, etc.

## Conda install and setup

## Python 3.12

- `conda env create -f environment-about-ai-torch-py312.yml`

- `pip install pandas`
- `pip install matplotlib`
- `pip install scikit-learn`
- `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
  > Only if Cuda 13 installed.

  > To check installed CUDA run:
  - `nvidia-smi`
  - `nvcc -V`
- `pip install sentencepiece`
- `pip install sacremoses`
- `pip install transformers datasets accelerate evaluate tokenizers`
- `pip install faiss-cpu hf_xet`

- bs4
- hf_xet
- langchain
- nltk
- PyPDF2
- requests
- scikit-learn
- sentence_transformers
- transformers

#### Web server

- "fastapi[standard]"

#### Frontend

- gradio
- streamlit
