#!/usr/bin/env python3
"""
Demonstrate the difference between training and evaluation modes with dropout.
"""

from mlx_lm import load
from embed import encode_texts
import numpy as np

# Load model
model, tokenizer = load("./gemma-3-270m-mlx", adapter_path="./adapters/best")

texts = ["Hello, world!", "This is a test."]

print("=== Testing without explicit mode setting ===")
print("(Model is in default mode - likely training mode)")

# Run multiple times to see if results are consistent
for i in range(3):
    embeddings = encode_texts(model, tokenizer, texts, "query", pooling="eos")
    print(f"Run {i+1}: First embedding norm: {np.linalg.norm(embeddings[0]):.6f}")

print("\n=== Testing with explicit training mode ===")
model.train()  # Force training mode
for i in range(3):
    embeddings = encode_texts(model, tokenizer, texts, "query", pooling="eos")
    print(f"Run {i+1}: First embedding norm: {np.linalg.norm(embeddings[0]):.6f}")

print("\n=== Testing with explicit evaluation mode ===")
model.eval()  # Force evaluation mode
for i in range(3):
    embeddings = encode_texts(model, tokenizer, texts, "query", pooling="eos")
    print(f"Run {i+1}: First embedding norm: {np.linalg.norm(embeddings[0]):.6f}")

print("\n=== Summary ===")
print("If dropout is working correctly:")
print("- Training mode: Results should vary between runs (due to dropout)")
print("- Evaluation mode: Results should be identical between runs (no dropout)")
print("- Default mode: Results depend on how the model was loaded")
