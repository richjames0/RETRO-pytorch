import math
import os
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from shutil import rmtree
from typing import Dict

import numpy as np


def is_true_env_flag(env_flag):
    return os.getenv(env_flag, "false").lower() in ("true", "1", "t")


def reset_folder_(p):
    path = Path(p)
    rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)


@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer


def check_key(key):
    if isinstance(key, tuple):
        raise NotImplementedError("BertEmbeds don't support tuple indexing")
    if isinstance(key, slice):
        if key.step:
            raise NotImplementedError("BertEmbeds don't support slice steps")
        if key.start < 0 or key.stop < 0:
            raise NotImplementedError("BertEmbeds don't support negative slices")


def parse_meta(entry: Dict):
    if "id" in entry:
        doc_id = entry["id"]
    elif "filename" in entry["meta"]:
        doc_id = entry["meta"]["filename"]
    else:
        raise Exception(f"Unknown Doc ID for: {entry}")
    return {"doc_id": doc_id, "pile_set_name": entry["meta"]["pile_set_name"]}
