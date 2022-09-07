from pathlib import Path

import jsonlines
import streamlit.components.v1 as components

DATA_DIR = Path("/datasets01/gptz_corpus_dedup_10_10_1_0.05_exp29/120321/")


def read_corpus(*, fold: str, partition: str, domain: str, limit: int = 100, offset: int = 0):
    docs = []
    with jsonlines.jsonlines.open(DATA_DIR / fold / partition / f"{domain}.jsonl") as f:
        for idx, row in enumerate(f):
            docs.append(row)
            if idx >= limit:
                break
    return docs


def text_with_scrollbar(text, height="450px"):
    components.html(f"""<div style="height:{height}; overflow: scroll;">{text}</div>""", height=500)


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
