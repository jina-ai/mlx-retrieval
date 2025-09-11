#!/usr/bin/env python3
"""
Get embeddings from a trained model.
"""

import mlx.core as mx
from bidirectional_gemma3 import patch_gemma3_for_bidirectional

patch_gemma3_for_bidirectional()

PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
BOS_TOKEN_ID = 2
SPECIAL_TOKEN_IDS = set([PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID])
# single query and document prompt
QUERY_PROMPT = "<unused0>"
DOC_PROMPT = "<unused1>"

# # qwen settings
# QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
# DOC_PROMPT = ""

print(f"QUERY_PROMPT: {QUERY_PROMPT}")
print(f"DOC_PROMPT: {DOC_PROMPT}")


def _mean_pooling(hidden_states, attention_mask):
    attention_mask = mx.expand_dims(attention_mask, axis=-1)
    masked_embeddings = hidden_states * attention_mask
    sum_embeddings = mx.sum(masked_embeddings, axis=1)
    sum_mask = mx.sum(attention_mask, axis=1)
    sum_mask = mx.where(sum_mask > 0, sum_mask, mx.ones_like(sum_mask))
    return sum_embeddings / sum_mask


def extract_embeddings(model, input_ids, attention_mask, normalize=True):
    # remove lm_head from model
    base_model = getattr(model, "model", model)
    # Set model to evaluation mode to disable dropout during inference
    base_model.eval()

    hidden_states = base_model(input_ids)
    embeddings = _mean_pooling(hidden_states, attention_mask)
    if normalize:
        embeddings = embeddings / mx.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def extract_eos_embeddings(
    model, input_ids, eos_positions, normalize=True, bidirectional=False
):
    # remove lm_head from model
    base_model = getattr(model, "model", model)
    # Set model to evaluation mode to disable dropout during inference
    base_model.eval()

    # Use bidirectional attention if requested
    hidden_states = base_model(input_ids, bidirectional=bidirectional)

    embeddings = hidden_states[mx.arange(hidden_states.shape[0]), eos_positions]
    if normalize:
        embeddings = embeddings / mx.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def encode_texts(
    model,
    tokenizer,
    texts,
    prompt_type,
    pooling="mean",
    max_length=256,
    normalize=True,
):
    if isinstance(texts, str):
        texts = [texts]

    input_ids, attention_mask, eos_positions = None, None, None
    # First pass: tokenize all texts and find max length in this batch
    all_tokens = []
    for text in texts:
        if prompt_type == "query":
            text = f"{QUERY_PROMPT}{text}<eos>"
        elif prompt_type == "document":
            text = f"{DOC_PROMPT}{text}<eos>"
        tokens = tokenizer.encode(text)[:max_length]
        all_tokens.append(tokens)

    # Find the maximum token length in this batch (capped by max_length)
    batch_max_length = min(max(len(tokens) for tokens in all_tokens), max_length)

    # Second pass: pad all texts to the batch max length and create batch tensors
    padded_tokens = []
    attention_masks = []
    eos_positions = []

    for tokens in all_tokens:
        if len(tokens) < batch_max_length:
            tokens += [PAD_TOKEN_ID] * (batch_max_length - len(tokens))
        padded_tokens.append(tokens)
        # get last <unused0> or <unused1> position
        last_eos_position = len(tokens) - 1 - tokens[::-1].index(EOS_TOKEN_ID)
        # everything after last_eos_position should be masked
        attention_masks.append(
            [0 if i > last_eos_position else 1 for i in range(len(tokens))]
        )
        if last_eos_position is None:
            raise ValueError("EOS token not found in tokens")
        eos_positions.append(last_eos_position)

    # Convert to batch tensors
    input_ids = mx.array(padded_tokens)
    attention_mask = mx.array(attention_masks)
    eos_positions = mx.array(eos_positions)

    # Extract embeddings for entire batch at once
    if pooling == "mean":
        embeddings = extract_embeddings(
            model, input_ids, attention_mask, normalize=normalize
        )
    elif pooling == "eos":
        embeddings = extract_eos_embeddings(
            model, input_ids, eos_positions, normalize=normalize
        )
    else:
        raise ValueError(f"Invalid pooling method: {pooling}")
    return embeddings
