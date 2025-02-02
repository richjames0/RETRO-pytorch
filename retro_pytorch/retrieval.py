import logging
import os
import pickle as pkl
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Optional

import faiss
import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from retro_pytorch.utils import memmap

LOG_KNNS_EVERY = 8_000  # under a minute on devfair
TRAINED_INDEX_SUFFIX = ".trained"
CASED = False
SOS_ID = 101
EOS_ID = 102
BERT_MODEL_DIM = 768
# TODO: this is very unpleasant
BERT_VOCAB_SIZE = 28996 if CASED else 30522

TMP_PATH = Path("./.tmp")
EMBEDDING_TMP_SUBFOLDER = "embeddings"

# helper functions


def exists(val):
    return val is not None


def range_chunked(max_value, *, batch_size):
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)
        counter = curr


# indexing helper functions


def faiss_read_index(path):
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)


# singleton globals

MODEL = None
TOKENIZER = None


def get_tokenizer(
    name: Optional[str] = None,
    repo_or_dir: str = "huggingface/pytorch-transformers",
    source: str = "github",
    skip_validation: bool = False,
):
    global TOKENIZER
    if name is None:
        name = f'bert-base-{"" if CASED else "un"}cased'
    if source == 'local':
        repo_or_dir = str(Path(repo_or_dir).expanduser())
    if not exists(TOKENIZER):
        TOKENIZER = torch.hub.load(repo_or_dir, "tokenizer", name, skip_validation=skip_validation, source=source)
    return TOKENIZER


def get_bert(
    name: Optional[str] = None,
    repo_or_dir: str = "huggingface/pytorch-transformers",
    source: str = "github",
    skip_validation: bool = False,
):
    if name is None:
        name = f'bert-base-{"" if CASED else "un"}cased'
    if source == 'local':
        repo_or_dir = str(Path(repo_or_dir).expanduser())
    global MODEL
    if not exists(MODEL):
        MODEL = torch.hub.load(
            repo_or_dir,
            "model",
            name,
            skip_validation=skip_validation,
            source=source,
        )
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()

    return MODEL


# tokenize


def tokenize(texts, name: Optional[str] = None, repo_or_dir: str = "huggingface/pytorch-transformers", source: str = "github", skip_validation: bool = False, add_special_tokens=True):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    tokenizer = get_tokenizer(name=name, repo_or_dir=repo_or_dir, source=source, skip_validation=skip_validation)
    encoding = tokenizer.batch_encode_plus(texts, add_special_tokens=add_special_tokens, padding=True, return_tensors="pt")
    token_ids = encoding.input_ids
    return token_ids


# text to chunks


def doc_text_to_chunks_and_seq_indices(*, doc_text, chunk_size=64, seq_len=2048, pad_id=0):
    assert (seq_len % chunk_size) == 0, "sequence length must be divisible by chunk size"

    # ignore spurious warning - we won't be feeding sequence into model as-is
    ids = tokenize(doc_text)
    ids = rearrange(ids, "1 ... -> ...")

    text_len = ids.shape[-1]

    # pad to multiple of chunk size with an extra token

    padding = chunk_size - ((text_len - 1) % chunk_size)
    ids = F.pad(ids, (0, padding))

    # split out very last token

    ids, last_token = ids[:-1], ids[-1:]
    assert last_token.item() == 0
    ids = rearrange(ids, "(n c) -> n c", c=chunk_size)

    # first tokens of chunk [2:] and on will become the last token of chunk [1:]

    last_token_per_chunk = ids[1:, 0]
    all_last_tokens = torch.cat((last_token_per_chunk, last_token), dim=0)
    all_last_tokens = rearrange(all_last_tokens, "n -> n 1")

    # append all last tokens to ids for (num_chunks, chunk_size + 1)

    chunks_with_extra_token = torch.cat((ids, all_last_tokens), dim=-1)

    # calculate chunk indices starting at 0, spaced number of chunks of seq len apart

    total_chunks = ids.shape[0]
    num_chunks_per_seq = seq_len // chunk_size
    seq = torch.arange(0, total_chunks, num_chunks_per_seq)

    return chunks_with_extra_token, seq


#####


def jsonl_folder_iterator(folder, glob):
    paths = sorted([*Path(folder).glob(glob)])
    for path in paths:
        with jsonlines.open(path) as reader:
            for obj in reader:
                yield obj["text"], path.name


# def jsonl_iterator(fname):
#     with jsonlines.open(fname) as reader:
#         for obj in reader:
#             yield obj['text']


def jsonl_folder_to_chunks_(
    folder,
    glob="**/*.jsonl",
    **kwargs,
):
    return _text_to_chunks(text_iterator=jsonl_folder_iterator(folder, glob), **kwargs)


#####


def text_folder_iterator(folder, glob):
    paths = sorted([*Path(folder).glob(glob)])
    for path in paths:
        yield path.read_text()


def text_folder_to_chunks_(
    folder,
    glob="**/*.txt",
    **text_to_chunks_kwargs,
):
    return _text_to_chunks(text_iterator=text_folder_iterator(folder, glob), **text_to_chunks_kwargs)


def _text_to_chunks(
    *,
    text_iterator,
    chunks_memmap_path,
    seqs_memmap_path,
    doc_ids_memmap_path,
    chunk_size=64,
    seq_len=2048,
    max_chunks=1_000_000,
    max_seqs=100_000,
):

    total_chunks = 0
    total_docs = 0
    total_seqs = 0

    chunks_shape = (max_chunks, chunk_size + 1)
    seqs_shape = (max_seqs,)
    doc_ids_shape = (max_chunks,)

    # TODO: docs for a given filename are a contiguous list so, with a bit more bookkeeping we could compress this structure
    filenames_to_docs = defaultdict(list)

    with memmap(chunks_memmap_path, shape=chunks_shape, dtype=np.int32, mode="w+") as chunks_memmap, memmap(
        seqs_memmap_path, shape=seqs_shape, dtype=np.int32, mode="w+"
    ) as seqs_memmap, memmap(doc_ids_memmap_path, shape=doc_ids_shape, dtype=np.int32, mode="w+") as doc_ids_memmap:
        logging.info("Processing docs (chunking, etc.)")

        for text, filename in text_iterator:
            chunks, seq = doc_text_to_chunks_and_seq_indices(doc_text=text, chunk_size=chunk_size, seq_len=seq_len)

            doc_chunk_len = chunks.shape[0]
            doc_seq_len = seq.shape[0]

            if doc_chunk_len > max_chunks - total_chunks:
                msg = f"Hit max chunks in {filename}; stopping"
                logging.error(msg)
                raise Exception(msg)
            if doc_seq_len > max_seqs - total_seqs:
                msg = f"Hit max seqs in {filename}; stopping"
                logging.error(msg)
                raise Exception(msg)

            chunks_memmap[total_chunks : (total_chunks + doc_chunk_len)] = chunks.numpy()
            seqs_memmap[total_seqs : (total_seqs + doc_seq_len)] = seq.numpy() + total_chunks
            # map each chunk to doc_id, having form 0, 0, ..., 1, 1,  ... 2, 2,
            doc_ids_memmap[total_chunks : (total_chunks + doc_chunk_len)] = np.full((doc_chunk_len,), total_docs)
            filenames_to_docs[filename].append(total_docs)

            total_chunks += doc_chunk_len
            total_seqs += doc_seq_len
            total_docs += 1

            if total_docs % 1000 == 0:
                logging.debug(f"Processed {total_docs} docs so far")

        logging.debug(f"Finished processing docs")

        # shuffle at one of last places before split into train, validation
        np.random.shuffle(seqs_memmap[0:(total_seqs)])

        with open(f"{doc_ids_memmap_path}.map", "wb") as f:
            pkl.dump(filenames_to_docs, f)

    return dict(chunks=total_chunks, docs=total_docs, seqs=total_seqs)


# embedding function


@torch.no_grad()
def bert_embed(token_ids, return_cls_repr=False, eps=1e-8, pad_id=0.0, model_config: Optional[DictConfig] = None):
    if model_config is None:
        model = get_bert()
    else:
        model = get_bert(**model_config)
    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    outputs = model(input_ids=token_ids, attention_mask=mask, output_hidden_states=True)

    hidden_state = outputs.hidden_states[-1]

    if return_cls_repr:
        return hidden_state[:, 0]  # return [cls] as representation

    if not exists(mask):
        return hidden_state.mean(dim=1)

    mask = mask[:, 1:]  # mean all tokens excluding [cls], accounting for length
    mask = rearrange(mask, "b n -> b n 1")

    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)
    return masked_mean


# chunks to knn


def chunks_to_embeddings_(
    *,
    num_chunks,
    chunks_memmap_path,
    embeddings_memmap_path,
    chunk_size=64,
    embed_dim=BERT_MODEL_DIM,
    batch_size=16,
    use_cls_repr=False,
    pad_id=0.0,
    worker_id=None,
    num_workers=1,
):
    chunks_shape = (num_chunks, chunk_size + 1)
    embed_shape = (num_chunks, embed_dim)

    # TODO: delete this and also num_workers code below unless choose to revive completely
    #       but note the latter would entail starting a bunch of processes which just call this function
    #       likely will take work to make this work since old code was broken and doesn't look good from
    #       the perspective of calling this directly either (e.g. when does the file ever get created)
    # if worker_id is not None:
    #     if not Path(embeddings_memmap_path + '.part1').exists():
    #         raise FileNotFoundError(f"When embedding with worker_ids, the numpy file must already exist to avoid accidental truncation of other worker output. Please create it by running from retro_pytorch.utils import BertEmbeds; BertEmbeds(fname = '{embeddings_memmap_path}', shape={embed_shape}, dtype=np.float32, mode='w+') before running any workers.")
    #     mode = 'r+'
    # else:
    mode = "w+"

    with memmap(chunks_memmap_path, shape=chunks_shape, dtype=np.int32) as chunks, memmap(
        embeddings_memmap_path, shape=embed_shape, dtype=np.float32, mode="w+"
    ) as embeddings:
        logging.info("Embedding chunks")

        for batch_index, dim_slice in enumerate(range_chunked(num_chunks, batch_size=batch_size)):
            # TODO: see todo above
            # If num_workers is 10, each worker does every 10'th batch (offset by worker_id)
            # if idx % num_workers != worker_id:
            #     continue

            batch_chunk_npy = chunks[dim_slice]

            batch_chunk = torch.from_numpy(batch_chunk_npy)

            cls_tokens = torch.full((batch_chunk.shape[0], 1), SOS_ID)
            batch_chunk = torch.cat((cls_tokens, batch_chunk), dim=1)

            batch_chunk = batch_chunk[
                :, :-1
            ]  # omit last token, the first token of the next chunk, used for autoregressive training

            batch_embed = bert_embed(batch_chunk, return_cls_repr=use_cls_repr)

            embeddings[dim_slice] = batch_embed.detach().cpu().numpy()
            for embed_of_batch_index, embedding in enumerate(embeddings[dim_slice]):
                if not embedding.any():
                    logging.error(f'Issue embedding ({batch_index},{embed_of_batch_index}): {chunks[batch_index * batch_size + embed_of_batch_index]} mapped to all zeros')
                    import pdb; pdb.set_trace()
            logging.debug(f"Embedded {dim_slice.stop} / {num_chunks}")

        logging.debug(f"Finished embedding chunks")


def train_faiss_index(embeddings, index_path):
    # Per https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    if embeddings.shape[0] < 1_000_000:
        num_clusters = int(sqrt(embeddings.shape[0]) * 8)
    elif embeddings.shape[0] > 1_000_000 and embeddings.shape[0] < 10_000_000:
        num_clusters = 65536
    elif embeddings.shape[0] > 10_000_000 and embeddings.shape[0] < 100_000_000:
        num_clusters = 262144
    else:
        num_clusters = 1048576
    num_train_examples = min(num_clusters * 64, embeddings.shape[0])
    logging.info(
        f"Training an IVF index with {num_clusters} clusters on {num_train_examples} training examples (out of {embeddings.shape[0]} total vectors)."
    )

    # This is the closest FAISS equivalent to Google's SCANN
    # See https://github.com/facebookresearch/faiss/wiki/The-index-factory for what this string means.
    # Note that the IndexIVFPQFastScan index created by this factory is intended for use on AVX2 enabled CPUs.
    # Also note that this index requires *at least* ~16x4 bits of RAM for every stored vector.
    # That's around 43 GB for 5.8B vectors.
    # TODO: Above comment is not correct given current key
    index = faiss.IndexFlatL2(BERT_MODEL_DIM)
    # index_factory(BERT_MODEL_DIM, f'IVF{num_clusters},PQ768')
    # index = faiss.index_factory(BERT_MODEL_DIM, f'OPQ16_64,IVF{num_clusters}_HNSW32,PQ16x4fs')
    # PCA256,L2norm,IVF1024,PCAR128,SHg')  #

    # From https://gist.github.com/mdouze/46d6bbbaabca0b9778fca37ed2bcccf6
    # For a sense of the speedup you get from using GPUs:
    # Clustering 3.5M points to 50k clusters takes about 2 min on a V100 and 20 min on AVX2.

    # FIXME: commenting out for temporary use of flat index
    # logging.info(f'Using {faiss.get_num_gpus()} gpus for index training')
    # index_ivf = faiss.extract_index_ivf(index)
    # clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
    # index_ivf.clustering_index = clustering_index

    # # Look at all the embeddings in the top 5 closest voronoi cells when doing the IVF lookup
    # # TODO (mitchg) - tinker with this later
    # index.nprobe = 5

    # if num_train_examples < embeddings.shape[0]:
    #     # This doesn't make a copy in RAM unless you cross the internal BERT filesize limit (10 TB)
    #     train_embeds = embeddings[0:num_train_examples]
    # else:
    #     train_embeds = embeddings

    # timer_logger = lambda x: None
    # logging.info('Training index')
    # with Timer('index training', logger=timer_logger):
    #     # Apparently FAISS plays nice with numpy memmap.
    #     index.train(train_embeds)
    # logging.info('Done training index')
    # logging.debug(Timer.timers)

    # Save the index immediately after training (which takes a long
    # time) in case we fail for some reason to add the embeds to the index.
    faiss.write_index(index, str(index_path) + TRAINED_INDEX_SUFFIX)

    return index


def index_embeddings(
    embeddings_path,
    embed_shape,
    index_path,
    faiss_num_threads=None,
):
    with memmap(embeddings_path, shape=embed_shape, dtype=np.float32, mode="r") as embeddings:
        pretrained_index_path = str(index_path) + TRAINED_INDEX_SUFFIX
        # we can do this because it's only created when the (long) process is completed
        if Path(pretrained_index_path).exists():
            index = faiss.read_index(pretrained_index_path)
        else:
            index = train_faiss_index(embeddings, index_path)

        if faiss_num_threads:
            # mitchg - I've had problems running at the default thread
            # count (10). Personally I've found 4 threads is the sweet
            # spot here, not sure why. Maybe some kind of thrashing at
            # higher threads?
            faiss.omp_set_num_threads(faiss_num_threads)

        num_embeds = embeddings.shape[0]
        logging.info(f"Adding {num_embeds} embeds to index")
        # Do it in chunks so we don't run out of RAM. For 5.8 B embeddings
        # and 4 threads, this took around 2 days.
        last_billion = 0
        for dim_slice in range_chunked(num_embeds, batch_size=128):
            if dim_slice.start % (128 * 1000) == 0:
                logging.debug(f"{dim_slice.start} / {num_embeds}")
                if dim_slice.start // 1_000_000_000 > last_billion:
                    last_billion += 1
                    logging.info(f"Writing index to {index_path} ({last_billion} billion)")
                    faiss.write_index(index, str(index_path))
            index.add(embeddings[dim_slice])

        logging.info(f"Writing index to {index_path}")
        faiss.write_index(index, str(index_path))
        logging.info(f"Index written")

        return index


def chunks_to_index_and_embed(
    *,
    num_chunks,
    chunk_size,
    embedding_path,
    chunk_memmap_path,
    use_cls_repr=False,
    max_rows_per_file=500,
    chunks_to_embeddings_batch_size=16,
    embed_dim=BERT_MODEL_DIM,
    index_path,
):
    embed_shape = (num_chunks, embed_dim)

    # TODO: this isn't ideal but creating embeddings takes so long it could be risky to assume presence of the file means all is well
    if os.environ.get("SKIP_EMBEDS") != "1":
        chunks_to_embeddings_(
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            chunks_memmap_path=chunk_memmap_path,
            embeddings_memmap_path=embedding_path,
            use_cls_repr=use_cls_repr,
            batch_size=chunks_to_embeddings_batch_size,
            embed_dim=embed_dim,
        )

    index = index_embeddings(
        embeddings_path=embedding_path,
        embed_shape=embed_shape,
        index_path=index_path,
    )

    embeddings = np.memmap(embedding_path, shape=embed_shape, dtype=np.float32, mode="r")
    return index, embeddings


def chunks_to_precalculated_knn_(
    *,
    num_nearest_neighbors,
    num_chunks,
    chunk_size,
    chunk_memmap_path,
    doc_ids_memmap_path,
    use_cls_repr=False,
    max_rows_per_file=500,
    chunks_to_embeddings_batch_size=16,
    embed_dim=BERT_MODEL_DIM,
    num_extra_neighbors=10,
    force_reprocess=False,
    index_path,
    **index_kwargs,
):
    # TODO: dont' reassign like this
    index_path = Path(index_path)

    chunk_path = Path(chunk_memmap_path)
    knn_path = chunk_path.parents[0] / f"{chunk_path.stem}.knn{chunk_path.suffix}"

    # early return knn path and faiss index
    # unless if force_reprocess is True

    if index_path.exists() and knn_path.exists() and not force_reprocess:
        logging.info(f"Preprocessed knns found at {str(knn_path)}, Faiss index reconstituted from {str(index_path)}")
        index = faiss_read_index(index_path)
        return knn_path, index

    # fetch the faiss index and calculated embeddings for the chunks

    embed_shape = (num_chunks, embed_dim)
    embedding_path = f"{chunk_memmap_path}.embedded"
    if index_path.exists() and Path(embedding_path).exists() and not force_reprocess:
        logging.info(
            f"Using precomputed Faiss index reconstituted from {str(index_path)}, embeddings found at {str(embedding_path)}"
        )
        index = faiss_read_index(index_path)
        embeddings = np.memmap(embedding_path, shape=embed_shape, dtype=np.float32, mode="r")
    else:
        index, embeddings = chunks_to_index_and_embed(
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            embedding_path=embedding_path,
            embed_dim=embed_dim,
            chunk_memmap_path=chunk_memmap_path,
            index_path=index_path,
            **index_kwargs,
        )

    total_neighbors_to_fetch = num_extra_neighbors + num_nearest_neighbors + 1

    with memmap(knn_path, shape=(num_chunks, num_nearest_neighbors), dtype=np.int32, mode="w+") as knns, memmap(
        doc_ids_memmap_path, shape=(num_chunks,), dtype=np.int32, mode="r"
    ) as doc_ids:
        logging.info("Calculating knns")

        for dim_slice in range_chunked(num_chunks, batch_size=max_rows_per_file):
            if ((dim_slice.start + max_rows_per_file) % LOG_KNNS_EVERY) == 0:
                logging.debug(
                    f"Calculating knns {dim_slice.start} / {num_chunks}"
                )  # TODO: THIS SHOULD BE A FUNCTION OF BATCH SIZE. 4k is 30 mins vs. 1 min 1k

            query_vector = embeddings[dim_slice]

            distances, indices = index.search(query_vector, k=total_neighbors_to_fetch)

            # remove self from distances and indices
            distances = distances[:, 1:]
            indices = indices[:, 1:]

            # mask out any neighbors that belong to the same document to -1
            query_doc_ids = doc_ids[dim_slice]
            neighbor_doc_ids = doc_ids[indices]
            neighbor_from_same_doc = query_doc_ids[..., None] == neighbor_doc_ids

            indices = np.where(neighbor_from_same_doc, -1, indices)
            distances = np.where(neighbor_from_same_doc, 1e3, distances)

            # re-sort indices by updated distances
            indices = np.take_along_axis(indices, np.argsort(distances, axis=1), axis=1)

            # store nearest neighbors to knn memmap
            knns[dim_slice] = indices[:, :num_nearest_neighbors]

    logging.info(f"Knns saved to {knn_path}")

    return knn_path, index
