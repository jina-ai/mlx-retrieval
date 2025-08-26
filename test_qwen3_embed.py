# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer
import numpy as np
from mlx_lm import load
import os
from tqdm import tqdm

# Configuration
DATA_VERSION = "v8"

# Load the Gemma tokenizer
print("Loading Gemma tokenizer...")
model_path = "./gemma-3-270m-mlx"
_, tokenizer = load(model_path)
print("Gemma tokenizer loaded successfully")

# Load the data
print(f"Loading {DATA_VERSION}.txt data...")
with open(f"data/{DATA_VERSION}.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(lines)} lines from {DATA_VERSION}.txt")

# Tokenize data line by line
print(f"Tokenizing {DATA_VERSION} data using Gemma tokenizer...")
tokenized = []
for line in tqdm(lines, desc="Tokenizing"):
    tokens = tokenizer.encode(line)
    tokenized.append(" ".join(map(str, tokens)))

# Save tokenized version
print("Saving tokenized data...")
with open(f"data/{DATA_VERSION}_tokenize.txt", "w", encoding="utf-8") as f:
    for tokenized_line in tokenized:
        f.write(tokenized_line + "\n")

print(f"Tokenized data saved to data/{DATA_VERSION}_tokenize.txt")
print(f"{DATA_VERSION}: {len(tokenized)} lines tokenized")

# Load the Qwen3 embedding model
print("Loading Qwen3 embedding model...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
print(f"Model loaded successfully. Max sequence length: {model.get_max_seq_length()}")

# Encode the documents with batch processing and progress tracking
print("Encoding documents with batch size 512...")
batch_size = 16
document_embeddings = []

for i in tqdm(range(0, len(lines), batch_size), desc="Encoding documents"):
    batch = lines[i : i + batch_size]
    batch_embeddings = model.encode(batch)
    document_embeddings.append(batch_embeddings)

# Concatenate all batches
document_embeddings = np.vstack(document_embeddings)
print(f"Document embeddings shape: {document_embeddings.shape}")

# Encode the same lines as queries with batch processing and progress tracking
print("Encoding queries with batch size 128...")
query_embeddings = []

for i in tqdm(range(0, len(lines), batch_size), desc="Encoding queries"):
    batch = lines[i : i + batch_size]
    batch_embeddings = model.encode(batch, prompt_name="query")
    query_embeddings.append(batch_embeddings)

# Concatenate all batches
query_embeddings = np.vstack(query_embeddings)
print(f"Query embeddings shape: {query_embeddings.shape}")

# Save the embeddings with compression
print("Saving embeddings with compression...")
np.savez_compressed(f"data/cal_{DATA_VERSION}_q.npz", embeddings=query_embeddings)
np.savez_compressed(f"data/cal_{DATA_VERSION}_d.npz", embeddings=document_embeddings)

print("Embeddings saved successfully:")
print(f"Query embeddings: data/cal_{DATA_VERSION}_q.npz ({query_embeddings.shape})")
print(
    f"Document embeddings: data/cal_{DATA_VERSION}_d.npz ({document_embeddings.shape})"
)

# Verify the saved files
print("\nVerification:")
print(
    f"Query embeddings file size: {os.path.getsize(f'data/cal_{DATA_VERSION}_q.npz') / (1024*1024):.2f} MB"
)
print(
    f"Document embeddings file size: {os.path.getsize(f'data/cal_{DATA_VERSION}_d.npz') / (1024*1024):.2f} MB"
)
