import json
import pickle
from pathlib import Path

import jsonlines
import numpy as np
import streamlit as st

from retro_pytorch.data import RETRODataset
from retro_pytorch.retrieval import get_tokenizer
from retro_pytorch.streamlit import DATA_DIR, text_with_scrollbar
from retro_pytorch.training import VALID_SPLIT

base_path = Path("/checkpoint/hcir/data/retro-z/derived_raw_00_sample_768")
stats_path = base_path / "processed-stats.json"
seqs_memmap_path = base_path / "train.seq.dat"
chunks_memmap_path = base_path / "train.chunks.dat"
docs_ids_memmap_path = base_path / "train.doc_ids.dat"

with open(stats_path) as f:
    stats = json.load(f)

num_chunks = stats["chunks"]
num_seqs = stats["seqs"]
chunk_size = 64
seq_len = 512
num_chunks_with_padding = num_chunks + seq_len // chunk_size

# num_chunks = seq_len // chunk_size
max_chunks = 5_000_000_000
max_seqs = 500_000_000

chunks_shape = (num_chunks_with_padding, chunk_size + 1)


with st.spinner("Loading data..."):
    if "filename_to_doc_id" in st.session_state:
        filename_to_doc_id = st.session_state["filename_to_doc_id"]
    else:
        with open(base_path / "train.doc_ids.dat.map", "rb") as f:
            filename_to_doc_id = pickle.load(f)
        st.session_state["filename_to_doc_id"] = filename_to_doc_id

    if "doc_id_to_filename_line" in st.session_state and "filename_to_max_line" in st.session_state:
        doc_id_to_filename_line = st.session_state["doc_id_to_filename_line"]
        filename_to_max_line = st.session_state["filename_to_max_line"]
    else:
        doc_id_to_filename_line = {}
        filename_to_max_line = {}
        for filename, lines in filename_to_doc_id.items():
            for idx, doc_id in enumerate(lines):
                doc_id_to_filename_line[doc_id] = (filename, idx)
            filename_to_max_line[filename] = max(lines)
        st.session_state["doc_id_to_filename_line"] = doc_id_to_filename_line
        st.session_state["filename_to_max_line"] = filename_to_max_line

    if "filename_line_to_content" in st.session_state:
        filename_line_to_content = st.session_state["filename_line_to_content"]
    else:
        filename_line_to_content = {}
        for filename in set(filename_to_doc_id.keys()):
            max_line_idx = filename_to_max_line[filename]
            with jsonlines.jsonlines.open(DATA_DIR / "train/00" / filename) as f:
                for idx, line in enumerate(f):
                    if idx > max_line_idx:
                        break
                    filename_line_to_content[(filename, idx)] = line
        st.session_state["filename_line_to_content"] = filename_line_to_content

    if "tokenizer" in st.session_state:
        tokenizer = st.session_state["tokenizer"]
    else:
        tokenizer = get_tokenizer()
        st.session_state["tokenizer"] = tokenizer

    num_valid_seqs = int(VALID_SPLIT * num_seqs)
    num_train_seqs = num_seqs - num_valid_seqs
    knn = 2

    if "dataset" in st.session_state:
        dataset = st.session_state["dataset"]
    else:
        dataset = RETRODataset(
            total_num_sequences=num_seqs,
            num_sequences=num_train_seqs,
            sequences_offset=0,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            seq_len=seq_len,
            chunk_memmap_path=chunks_memmap_path,
            chunk_nn_memmap_path=base_path / "train.chunks.knn.dat",
            seq_memmap_path=seqs_memmap_path,
            retrieve=True,
            num_neighbors=knn,
        )
        st.session_state["dataset"] = dataset

    if "chunks" in st.session_state:
        chunks = st.session_state["chunks"]
    else:
        chunks = np.memmap(chunks_memmap_path, shape=chunks_shape, dtype=np.int32, mode="r")
        st.session_state["chunks"] = chunks

    if "seqs" in st.session_state:
        seqs = st.session_state["seqs"]
    else:
        seqs = np.memmap(seqs_memmap_path, shape=(num_seqs,), dtype=np.int32, mode="r")
        st.session_state["seqs"] = seqs

    if "doc_ids" in st.session_state:
        doc_ids = st.session_state["doc_ids"]
    else:
        doc_ids = np.memmap(docs_ids_memmap_path, shape=(num_chunks_with_padding,), dtype=np.int32, mode="r")
        st.session_state["doc_ids"] = doc_ids


def token_ids_to_tokens(arr):
    return [tokenizer.convert_ids_to_tokens(t) for t in arr.tolist()]


def array_to_text(arr):
    return " ".join(token_ids_to_tokens(arr))


def print_text(text):
    st.markdown(f"<p>{text}</p>", unsafe_allow_html=True)


n_dataset = len(dataset)
st.markdown(
    f"""

# Retro Training Input

Number of Training Examples: {n_dataset}
"""
)
example_idx = int(st.number_input("Training Example Index", min_value=0, max_value=n_dataset - 1, value=0, step=1))

example, retrieved = dataset[example_idx]
begin_chunk_idx = seqs[example_idx]
doc_id = doc_ids[begin_chunk_idx]
filename, line = doc_id_to_filename_line[doc_id]
content = filename_line_to_content[(filename, line)]


st.markdown(
    f"""## Model Input/Prompt
- Begin Chunk IDX: {begin_chunk_idx}
- Doc IDX: {doc_id}
- Filename: {filename}
- Line: {line}
"""
)

st.markdown("## Text from JSONL File")
st.json(content, expanded=False)
# text_with_scrollbar(content, height="200px")
example_text = array_to_text(example)

st.markdown("## Text as Model Sees It")
st.write(example_text)

# Ensure dims are: chunks x neighbors x tokens
assert retrieved.shape[0] == 8
assert retrieved.shape[1] == 2

n_chunks = seq_len // chunk_size

st.markdown("## Nearest neighbors to each input chunk")
for i in range(n_chunks - 1):
    st.markdown(f"### Chunk {i}")
    chunk_tokens = token_ids_to_tokens(example[i * chunk_size : (i + 1) * chunk_size])
    unique_tokens = set(chunk_tokens)
    if len(unique_tokens) == 1 and "[PAD]" in unique_tokens:
        st.warning("Chunk is empty/only [PAD] tokens, skipping...")
        continue
    st.info(" ".join(chunk_tokens))
    for j in range(knn):
        st.markdown(f"#### Neighbor {j}")
        print_text(array_to_text(retrieved[i][j]))
