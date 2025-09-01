#!/usr/bin/env python3

import mlx.data as dx
import numpy as np
import random


# Global cache for loaded data
_CACHE = {}


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
        query_tokens = [tokens[0], 6] + tokens[1:] + [1]  # [...content, role, eos]
        doc_tokens = [tokens[0], 7] + tokens[1:] + [1]  # [bos, ...content, role, eos]
        tokenized_arrays.append(
            {
                "query": np.array(query_tokens, dtype=np.int32),
                "doc": np.array(doc_tokens, dtype=np.int32),
                "eos_pos": len(query_tokens) - 1,  # bos, role, content, eos
            }
        )

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
            "eos_pos": tokenized_arrays[idx]["eos_pos"],
        }

        if is_query:
            sample["tokenized"] = tokenized_arrays[idx]["query"]
            sample["embedding"] = query_embeddings[idx].astype(np.float32)
        else:
            sample["tokenized"] = tokenized_arrays[idx]["doc"]
            sample["embedding"] = doc_embeddings[idx].astype(np.float32)
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
    stream = stream.dynamic_batch(1000, "tokenized", max_data_size=batch_size, num_threads=4, shuffle=True)
    return stream
