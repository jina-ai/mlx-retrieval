#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, Generator, Iterable, List, Optional
import mlx.data as dx
from elasticsearch import Elasticsearch
from embed import PAD_TOKEN_ID, SPECIAL_TOKEN_IDS


# Get required environment variables
ES_NODE = os.environ.get("ES_NODE")
ES_INDEX = os.environ.get("ES_INDEX")
ES_API_KEY = os.environ.get("ES_API_KEY")

# Global Elasticsearch client (only initialized if ES variables are set)
_es_client = None
if ES_NODE and ES_INDEX and ES_API_KEY:
    _es_client = Elasticsearch(
        [ES_NODE],
        api_key=ES_API_KEY,
        max_retries=3,
        retry_on_timeout=True,
    )


def _make_positive_doc(doc_source: Dict[str, object]) -> Optional[str]:
    title = doc_source.get("title")
    snippet = doc_source.get("snippet")

    title_str = title if isinstance(title, str) else None
    snippet_str = snippet if isinstance(snippet, str) else None

    if title_str and snippet_str:
        return f"{title_str}\n{snippet_str}"
    if snippet_str:
        return snippet_str
    if title_str:
        return title_str
    return None


def _iter_es_pairs_random_sampler(
    index: str = ES_INDEX,
    batch_size: int = 64,
    probability: float = 1e-6,
    seed: Optional[int] = 42,
    source_fields: Iterable[str] = ("query", "snippet", "title", "rank"),
) -> Generator[List[Dict[str, object]], None, None]:
    if _es_client is None:
        raise ValueError(
            "Elasticsearch client not initialized. Please set ES_NODE, ES_INDEX, and ES_API_KEY environment variables."
        )

    client = _es_client

    current_seed = seed if seed is not None else 42

    while True:
        body = {
            "size": 0,
            "track_total_hits": False,
            "aggregations": {
                "sampling": {
                    "random_sampler": {
                        "probability": probability,
                        "seed": current_seed,
                    },
                    "aggs": {
                        "docs": {
                            "top_hits": {
                                "size": min(batch_size * 2, 100),
                                "_source": list(source_fields),
                            }
                        }
                    },
                }
            },
        }

        try:
            resp = client.search(index=index, body=body)
        except Exception as e:
            print(f"Elasticsearch error: {e}")
            time.sleep(1)
            continue

        docs = resp["aggregations"]["sampling"]["docs"]["hits"]["hits"]
        if not docs:
            current_seed += 1
            continue

        batch = []
        for doc in docs:
            source = doc["_source"]
            query = source.get("query")
            positive = _make_positive_doc(source)
            rank = source.get("rank", 1)

            if query and positive:
                batch.append({"query": query, "positive": positive, "rank": rank})

        if batch:
            yield batch

        current_seed += 1


def _iter_local_jsonl_pairs(
    file_path: str = "train-data.jsonl",
    batch_size: int = 64,
    seed: Optional[int] = None,
) -> Generator[List[Dict[str, object]], None, None]:
    """Iterator for local JSONL files with same API as _iter_es_pairs_random_sampler"""
    # Read all data into memory
    all_items = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    query = item.get("query")
                    document = item.get("document")
                    rank = item.get("rank", 1)

                    if query and document:
                        all_items.append(
                            {"query": query, "positive": document, "rank": rank}
                        )
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return

    if not all_items:
        print(f"Warning: No valid items found in {file_path}")
        return

    # Yield batches
    while True:
        for i in range(0, len(all_items), batch_size):
            batch = all_items[i : i + batch_size]
            if batch:
                yield batch


def training_batch_stream(
    batch_size: int = 64,
    prefetch_batches: int = 8,
    num_threads: int = 4,
    max_length: int = 256,
    tokenizer=None,
    use_local_data: bool = True,
    local_file_path: str = "train-data.jsonl",
) -> dx.Stream:
    """Create MLX Data stream that outputs training-ready batches directly"""

    if tokenizer is None:
        raise ValueError("Tokenizer is required")

    def training_batch_generator():
        current_batch = []

        if use_local_data:
            data_iterator = _iter_local_jsonl_pairs(
                file_path=local_file_path, batch_size=batch_size
            )
        else:
            data_iterator = _iter_es_pairs_random_sampler(batch_size=batch_size)

        for batch in data_iterator:
            for item in batch:

                # Decode bytes to strings
                query = (
                    item["query"].decode("utf-8", errors="ignore")
                    if isinstance(item["query"], bytes)
                    else item["query"]
                )
                positive = (
                    item["positive"].decode("utf-8", errors="ignore")
                    if isinstance(item["positive"], bytes)
                    else item["positive"]
                )
                rank = int(item["rank"])

                current_batch.append(
                    {"query": query, "positive": positive, "rank": rank}
                )

                if len(current_batch) == batch_size:
                    # Create training-ready batch
                    training_batch = _create_training_batch_from_texts(
                        current_batch,
                        tokenizer,
                        max_length,
                    )
                    yield training_batch
                    current_batch = []

        # Yield remaining samples if any
        if current_batch:
            training_batch = _create_training_batch_from_texts(
                current_batch,
                tokenizer,
                max_length,
            )
            yield training_batch

    # Create MLX Data stream from the generator
    stream = dx.stream_python_iterable(training_batch_generator)

    # Apply optimizations
    stream = stream.prefetch(prefetch_batches, num_threads)

    return stream


def _create_training_batch_from_texts(
    batch_items,
    tokenizer,
    max_length: int,
):
    """Helper function to create training batch from text items"""

    # First pass: tokenize all items and find max lengths in this batch
    raw_query_tokens, raw_doc_tokens = [], []
    ranks = []

    for item in batch_items:
        query = item["query"]
        positive = item["positive"]
        rank = item["rank"]

        q_tokens = tokenizer.encode(f"<unused0>{query}<eos>")[:max_length]
        d_tokens = tokenizer.encode(f"<unused1>{positive}<eos>")[:max_length]

        raw_query_tokens.append(q_tokens)
        raw_doc_tokens.append(d_tokens)
        ranks.append(rank)

    # Find the maximum token lengths in this batch (capped by max_length)
    batch_query_max_length = min(
        max(len(tokens) for tokens in raw_query_tokens), max_length
    )
    batch_doc_max_length = min(
        max(len(tokens) for tokens in raw_doc_tokens), max_length
    )

    # Second pass: pad all items to the batch max lengths
    query_tokens, doc_tokens = [], []
    query_masks, doc_masks = [], []

    for q_tokens, d_tokens in zip(raw_query_tokens, raw_doc_tokens):
        # Pad queries to batch query max length
        if len(q_tokens) < batch_query_max_length:
            q_tokens = q_tokens + [PAD_TOKEN_ID] * (
                batch_query_max_length - len(q_tokens)
            )

        # no padding, bos, eos
        q_mask = [1 if t not in SPECIAL_TOKEN_IDS else 0 for t in q_tokens]

        # Pad docs to batch doc max length
        if len(d_tokens) < batch_doc_max_length:
            d_tokens = d_tokens + [PAD_TOKEN_ID] * (
                batch_doc_max_length - len(d_tokens)
            )
        d_mask = [1 if t not in SPECIAL_TOKEN_IDS else 0 for t in d_tokens]

        query_tokens.append(q_tokens)
        doc_tokens.append(d_tokens)
        query_masks.append(q_mask)
        doc_masks.append(d_mask)

    # Calculate weights from ranks
    weights = []
    for r in ranks:
        r_int = r if isinstance(r, int) else 1
        if r_int <= 0:
            r_int = 1
        denom = math.log2(1.0 + float(r_int))
        w = 1.0 / denom if denom > 0 else 1.0
        weights.append(w)

    return {
        "query_input_ids": query_tokens,
        "doc_input_ids": doc_tokens,
        "query_attention_mask": query_masks,
        "doc_attention_mask": doc_masks,
        "weights": weights,
    }
