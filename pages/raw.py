import glob
from pathlib import Path
from typing import Dict

import jsonlines
import streamlit as st
import streamlit.components.v1 as components
from annotated_text import annotated_text

from retro_pytorch.retrieval import get_tokenizer
from retro_pytorch.streamlit import (
    DATA_DIR,
    DOMAINS,
    FOLDS,
    read_corpus,
    text_with_scrollbar,
)
from retro_pytorch.utils import parse_meta

st.title("Retro Dataset Viewer")
with st.sidebar:
    lines_to_load = st.number_input("Number of Documents to Load", 100)
    max_tokens = st.number_input("Max Tokens", 512)
    fold = st.selectbox("Fold", FOLDS)
    fold_partitions = [p.split("/")[-1] for p in glob.glob(str(DATA_DIR / fold / "*"))]
    partition = st.selectbox("Partition", fold_partitions)
    domain = st.selectbox("Domain", DOMAINS)

tokenizer = get_tokenizer()

docs_to_explore = read_corpus(fold=fold, partition=partition, domain=domain, limit=lines_to_load)

doc_idx = int(st.sidebar.number_input("Doc IDX", min_value=0, max_value=len(docs_to_explore) - 1))

document = docs_to_explore[doc_idx]

st.subheader("Document Metadata")
st.text(parse_meta(document))

st.subheader("Document Text")
text_with_scrollbar(document["text"])

st.subheader("Tokenized Text")
tokens = [(t, f"{idx}") for idx, t in enumerate(tokenizer.tokenize(document["text"]))][:max_tokens]
# st.code(tokens)

annotated_text(*tokens)
