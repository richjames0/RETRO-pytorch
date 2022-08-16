import glob
from pathlib import Path
from typing import Dict

import jsonlines
import streamlit as st
import streamlit.components.v1 as components
from annotated_text import annotated_text

from retro_pytorch.retrieval import get_tokenizer


DATA_DIR = Path("/datasets01/gptz_corpus_dedup_10_10_1_0.05_exp29/120321/")

FILE_HANDLES = {}


def read_corpus(*, fold: str, partition: str, domain: str, limit: int = 100, offset: int = 0):
    docs = []
    with jsonlines.jsonlines.open(DATA_DIR / fold / partition / f"{domain}.jsonl") as f:
        for idx, row in enumerate(f):
            docs.append(row)
            if idx >= limit:
                break
    return docs


def parse_meta(entry: Dict):
    if "id" in entry:
        doc_id = entry["id"]
    elif "filename" in entry["meta"]:
        doc_id = entry["meta"]["filename"]
    else:
        raise Exception(f"Unknown Doc ID for: {entry}")
    return {"doc_id": doc_id, "pile_set_name": entry["meta"]["pile_set_name"]}


def text_with_scrollbar(text):
    components.html(f"""<div style="height:450px; overflow: scroll;">{text}</div>""", height=500)


DOMAINS = [
    "BookCorpusFair",
    "ccnewsv2",
    "CommonCrawl",
    "DM_Mathematics",
    "Enron_Emails",
    "Gutenberg_PG-19",
    "HackerNews",
    "OpenSubtitles",
    "OpenWebText2",
    "redditflattened",
    "stories",
    "USPTO",
    "Wikipedia_en",
]
FOLDS = ["train", "valid"]

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
