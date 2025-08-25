#!/usr/bin/env python3

import mlx.data as dx
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from faker import Faker

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
fake = Faker(["zh_CN", "en_US"])
Faker.seed(4321)


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


def sample_generator(tokenizer, num_samples):
    """Generate samples with random role assignment"""
    for _ in range(num_samples):
        yield fake.sentence(nb_words=100)


def get_cali_stream(tokenizer, num_samples=10000, batch_size=4):
    """Create MLX Data stream with shuffling and dynamic batching"""

    stream = dx.stream_python_iterable(
        lambda: sample_generator(tokenizer, num_samples)
    )
    stream = stream.dynamic_batch(batch_size, "tokenized")
    return stream
