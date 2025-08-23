#!/usr/bin/env python3

import mlx.data as dx
import numpy as np
import random


# Global cache for loaded data
_CACHE = {}


def load_cali_data(version="v5"):
    """Load embeddings and tokenized data with caching"""
    if version in _CACHE:
        return _CACHE[version]

    if version == "v5":
        query_embeddings = np.load("data/cal_v5_q.npy")
        doc_embeddings = np.load("data/cal_v5_d.npy")
        tokenized_file = "cal_v5_tokenize.txt"
    else:
        query_embeddings = np.load("data/cal_v3_q.npy")
        doc_embeddings = np.load("data/cal_v3_d.npy")
        tokenized_file = "cal_v3_tokenize.txt"

    # Pre-process all tokenized lines to numpy arrays
    with open(tokenized_file, "r") as f:
        raw_lines = [line.strip() for line in f.readlines()]

    # Pre-allocate arrays for efficiency
    tokenized_arrays = []
    for line in raw_lines:
        tokens = [int(x) for x in line.split()]
        # Pre-insert role tokens and EOS for both query/doc variants
        query_tokens = [tokens[0], 6] + tokens[1:] + [1]  # [bos, role, ...content, eos]
        doc_tokens = [tokens[0], 7] + tokens[1:] + [1]  # [bos, role, ...content, eos]
        tokenized_arrays.append(
            {
                "query": np.array(query_tokens, dtype=np.int32),
                "doc": np.array(doc_tokens, dtype=np.int32),
            }
        )

    _CACHE[version] = (query_embeddings, doc_embeddings, tokenized_arrays)
    return _CACHE[version]


def create_sample(line_idx, tokenized_arrays, query_embeddings, doc_embeddings):
    """Create a single sample with random role assignment"""
    is_query = random.choice([True, False])

    # exclude bos and eos
    attention_mask = [0] + [1] * (len(tokenized_arrays[line_idx]["query"]) - 2) + [0]
    if is_query:
        return {
            "tokenized": tokenized_arrays[line_idx]["query"],
            "embedding": query_embeddings[line_idx].astype(np.float32),
            "idx": np.array(line_idx, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
        }
    else:
        return {
            "tokenized": tokenized_arrays[line_idx]["doc"],
            "embedding": doc_embeddings[line_idx].astype(np.float32),
            "idx": np.array(line_idx, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
        }


def sample_generator(tokenized_arrays, query_embeddings, doc_embeddings):
    """Generate samples with random role assignment"""
    indices = list(range(len(tokenized_arrays)))
    random.shuffle(indices)
    for idx in indices:
        yield create_sample(idx, tokenized_arrays, query_embeddings, doc_embeddings)


def get_cali_stream(version="v5", batch_size=4):
    """Create MLX Data stream with shuffling and dynamic batching"""
    query_embeddings, doc_embeddings, tokenized_arrays = load_cali_data(version)

    stream = dx.stream_python_iterable(
        lambda: sample_generator(tokenized_arrays, query_embeddings, doc_embeddings)
    )
    stream = stream.dynamic_batch(batch_size, "tokenized")
    return stream
