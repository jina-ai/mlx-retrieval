#!/usr/bin/env python3
"""
Test the real impact of dropout by using different inputs and measuring variance.
"""

from mlx_lm import load
from embed import encode_texts
import numpy as np

# Load model
model, tokenizer = load("./gemma-3-270m-mlx", adapter_path="./adapters/best")

# Different input texts to trigger different dropout patterns
texts1 = ["Hello world", "Test sentence"]
texts2 = ["Different input", "Another test"]

print("=== Testing Dropout Impact ===")
print("Using different input texts to trigger different dropout patterns\n")

# Test 1: Same input multiple times (should be identical in eval mode)
print("Test 1: Same input, multiple runs")
print("Input: ['Hello world', 'Test sentence']")

model.eval()  # Evaluation mode
embeddings1_eval = encode_texts(model, tokenizer, texts1, "query", pooling="eos")
embeddings2_eval = encode_texts(model, tokenizer, texts1, "query", pooling="eos")
diff_eval = np.mean(np.abs(embeddings1_eval - embeddings2_eval))
print(f"  Eval mode - Mean difference: {diff_eval:.8f}")

model.train()  # Training mode
embeddings1_train = encode_texts(model, tokenizer, texts1, "query", pooling="eos")
embeddings2_train = encode_texts(model, tokenizer, texts1, "query", pooling="eos")
diff_train = np.mean(np.abs(embeddings1_train - embeddings2_train))
print(f"  Train mode - Mean difference: {diff_train:.8f}")

# Test 2: Different inputs (should be different regardless of mode)
print("\nTest 2: Different inputs")
print("Input 1: ['Hello world', 'Test sentence']")
print("Input 2: ['Different input', 'Another test']")

model.eval()
embeddings1_eval = encode_texts(model, tokenizer, texts1, "query", pooling="eos")
embeddings2_eval = encode_texts(model, tokenizer, texts2, "query", pooling="eos")
diff_inputs_eval = np.mean(np.abs(embeddings1_eval - embeddings2_eval))
print(f"  Eval mode - Mean difference: {diff_inputs_eval:.8f}")

model.train()
embeddings1_train = encode_texts(model, tokenizer, texts1, "query", pooling="eos")
embeddings2_train = encode_texts(model, tokenizer, texts2, "query", pooling="eos")
diff_inputs_train = np.mean(np.abs(embeddings1_train - embeddings2_train))
print(f"  Train mode - Mean difference: {diff_inputs_train:.8f}")

print("\n=== Analysis ===")
print(f"Same input, eval mode: {diff_eval:.8f} (should be ~0.0)")
print(f"Same input, train mode: {diff_train:.8f} (may be >0.0 due to dropout)")
print(f"Different inputs, eval mode: {diff_inputs_eval:.8f} (should be >0.0)")
print(f"Different inputs, train mode: {diff_inputs_train:.8f} (should be >0.0)")

if diff_eval < 1e-8:
    print("✅ Eval mode working: Same input gives identical results")
else:
    print("❌ Eval mode issue: Same input gives different results")

if diff_train > 1e-8:
    print("✅ Train mode working: Dropout is active")
else:
    print("❌ Train mode issue: Dropout not working")
