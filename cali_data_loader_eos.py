#!/usr/bin/env python3

import mlx.data as dx
import numpy as np
import random
from embed import EOS_TOKEN_ID


# Global cache for loaded data
_CACHE = {}
# single special token setup
QUERY_PREFIX = [2, 6]
DOC_PREFIX = [2, 7]

# # qwen3 settings
# QUERY_PREFIX = [
#     2,
#     218875,
#     236787,
#     17770,
#     496,
#     4108,
#     3927,
#     7609,
#     236764,
#     33205,
#     7798,
#     42437,
#     600,
#     3890,
#     506,
#     7609,
#     107,
#     7990,
#     236787,
# ]
# DOC_PREFIX = [2]

print(f"QUERY_PREFIX: {QUERY_PREFIX}")
print(f"DOC_PREFIX: {DOC_PREFIX}")


def load_cali_data(version="v6"):
    """Load embeddings and tokenized data with caching"""
    if version in _CACHE:
        return _CACHE[version]

    # Dynamic file paths based on version parameter
    query_embeddings = np.load(f"data/cal_{version}_q.npz")["embeddings"][:, :640]
    doc_embeddings = np.load(f"data/cal_{version}_d.npz")["embeddings"][:, :640]
    query_embeddings = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )
    doc_embeddings = doc_embeddings / np.linalg.norm(
        doc_embeddings, axis=1, keepdims=True
    )
    tokenized_file = f"data/{version}_tokenize.txt"

    # Pre-process all tokenized lines to numpy arrays
    with open(tokenized_file, "r") as f:
        raw_lines = [line.strip() for line in f.readlines()]

    # Pre-allocate arrays for efficiency
    tokenized_arrays = []
    for line in raw_lines:
        tokens = [int(x) for x in line.split()]
        # Pre-insert role tokens and EOS for both query/doc variants
        query_tokens = (
            QUERY_PREFIX + tokens[1:] + [1]
        )  # [prefix, ...content, role, eos]
        doc_tokens = DOC_PREFIX + tokens[1:] + [1]  # [prefix, ...content, role, eos]
        query_last_eos_position = (
            len(query_tokens) - 1 - query_tokens[::-1].index(EOS_TOKEN_ID)
        )
        doc_last_eos_position = (
            len(doc_tokens) - 1 - doc_tokens[::-1].index(EOS_TOKEN_ID)
        )
        tokenized_arrays.append(
            {
                "query": np.array(query_tokens, dtype=np.int32),
                "doc": np.array(doc_tokens, dtype=np.int32),
                "eos_pos_query": query_last_eos_position,
                "eos_pos_doc": doc_last_eos_position,
            }
        )
    # check if len(tokenized_arrays) is same as len(query_embeddings) and len(doc_embeddings)
    if (
        len(tokenized_arrays) != query_embeddings.shape[0]
        or len(tokenized_arrays) != doc_embeddings.shape[0]
    ):
        raise ValueError(
            f"Length mismatch: {len(tokenized_arrays)} != {query_embeddings.shape[0]} != {doc_embeddings.shape[0]}"
        )
    print(f"Number of samples: {len(tokenized_arrays)}")

    _CACHE[version] = (query_embeddings, doc_embeddings, tokenized_arrays)
    return _CACHE[version]


def sample_generator(tokenized_arrays, query_embeddings, doc_embeddings):
    """Generate samples with random role assignment"""
    indices = list(range(len(tokenized_arrays)))
    is_queries = np.random.randint(0, 2, len(indices))
    random.shuffle(indices)
    for idx in indices:
        is_query = is_queries[idx]
        sample = {
            "idx": np.array(idx, dtype=np.int32),
        }

        if is_query:
            sample["tokenized"] = tokenized_arrays[idx]["query"]
            sample["embedding"] = query_embeddings[idx].astype(np.float32)
            sample["eos_pos"] = tokenized_arrays[idx]["eos_pos_query"]
        else:
            sample["tokenized"] = tokenized_arrays[idx]["doc"]
            sample["embedding"] = doc_embeddings[idx].astype(np.float32)
            sample["eos_pos"] = tokenized_arrays[idx]["eos_pos_doc"]
        yield sample


def get_cali_stream(version="v6", batch_size=10000):
    """Create MLX Data stream with shuffling and dynamic batching"""
    if isinstance(version, str):
        version = [version]

    query_embeddings = []
    doc_embeddings = []
    tokenized_arrays = []

    for v in version:
        qe, de, ta = load_cali_data(v)
        query_embeddings.append(qe)
        doc_embeddings.append(de)
        tokenized_arrays.append(ta)

    query_embeddings = np.concatenate(query_embeddings, axis=0)
    doc_embeddings = np.concatenate(doc_embeddings, axis=0)
    tokenized_arrays = np.concatenate(tokenized_arrays, axis=0)

    stream = dx.stream_python_iterable(
        lambda: sample_generator(tokenized_arrays, query_embeddings, doc_embeddings)
    )
    stream = stream.dynamic_batch(
        1000, "tokenized", max_data_size=batch_size, num_threads=4, shuffle=True
    )
    return stream
