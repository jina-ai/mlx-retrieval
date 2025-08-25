# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer
import numpy as np
from mlx_lm import load
import os
from tqdm import tqdm

# Load the Gemma tokenizer
print("Loading Gemma tokenizer...")
model_path = "./gemma-3-270m-mlx"
_, tokenizer = load(model_path)
print("Gemma tokenizer loaded successfully")

# Load the v7.txt data
print("Loading v7.txt data...")
with open("data/v7.txt", "r", encoding="utf-8") as f:
    v7_lines = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(v7_lines)} lines from v7.txt")

# Tokenize v7 data line by line
print("Tokenizing v7 data using Gemma tokenizer...")
v7_tokenized = []
for line in tqdm(v7_lines, desc="Tokenizing"):
    tokens = tokenizer.encode(line)
    v7_tokenized.append(" ".join(map(str, tokens)))

# Save tokenized version
print("Saving tokenized data...")
with open("data/v7_tokenize.txt", "w", encoding="utf-8") as f:
    for tokenized_line in v7_tokenized:
        f.write(tokenized_line + "\n")

print(f"Tokenized data saved to data/v7_tokenize.txt")
print(f"v7: {len(v7_tokenized)} lines tokenized")

# Load the Qwen3 embedding model
print("Loading Qwen3 embedding model...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
print(f"Model loaded successfully. Max sequence length: {model.get_max_seq_length()}")

# Encode the documents with batch processing and progress tracking
print("Encoding documents with batch size 512...")
batch_size = 512
document_embeddings = []

for i in tqdm(range(0, len(v7_lines), batch_size), desc="Encoding documents"):
    batch = v7_lines[i:i + batch_size]
    batch_embeddings = model.encode(batch)
    document_embeddings.append(batch_embeddings)

# Concatenate all batches
document_embeddings = np.vstack(document_embeddings)
print(f"Document embeddings shape: {document_embeddings.shape}")

# Encode the same lines as queries with batch processing and progress tracking
print("Encoding queries with batch size 128...")
query_embeddings = []

for i in tqdm(range(0, len(v7_lines), batch_size), desc="Encoding queries"):
    batch = v7_lines[i:i + batch_size]
    batch_embeddings = model.encode(batch, prompt_name="query")
    query_embeddings.append(batch_embeddings)

# Concatenate all batches
query_embeddings = np.vstack(query_embeddings)
print(f"Query embeddings shape: {query_embeddings.shape}")

# Save the embeddings with compression
print("Saving embeddings with compression...")
np.savez_compressed("data/cal_v7_q.npz", embeddings=query_embeddings)
np.savez_compressed("data/cal_v7_d.npz", embeddings=document_embeddings)

print("Embeddings saved successfully:")
print(f"Query embeddings: data/cal_v7_q.npz ({query_embeddings.shape})")
print(f"Document embeddings: data/cal_v7_d.npz ({document_embeddings.shape})")

# Verify the saved files
print("\nVerification:")
print(f"Query embeddings file size: {os.path.getsize('data/cal_v7_q.npz') / (1024*1024):.2f} MB")
print(f"Document embeddings file size: {os.path.getsize('data/cal_v7_d.npz') / (1024*1024):.2f} MB")
