from mlx_lm import load
from embed import encode_texts

model, tokenizer = load("./gemma-3-270m-mlx", adapter_path="./adapters/step_45000")

texts = [
    "Hello, world!",
    "This is a test.",
    "see.",
]

embeddings = encode_texts(model, tokenizer, texts, "query", pooling="eos")
print(f"Embeddings dtype: {embeddings.dtype}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")
print(f"Sample values precision: {embeddings[0][0]:.6f}")
