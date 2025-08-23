# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer
import numpy as np
from mlx_lm import load

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
print(model.get_max_seq_length())
exit()

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-0.6B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)


# Load the Gemma tokenizer
model_path = "./gemma-3-270m-mlx"
_, tokenizer = load(model_path)

# Load the calibration data v3
with open("data/calibration_datav3.txt", "r", encoding="utf-8") as f:
    v3_lines = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(v3_lines)} lines from calibration data v3")

# Load the calibration data v5
with open("data/calibration_data_v5_rc.txt", "r", encoding="utf-8") as f:
    v5_lines = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(v5_lines)} lines from calibration data v5")

# Tokenize v3 data
print("Tokenizing v3 data...")
v3_tokenized = []
for line in v3_lines:
    tokens = tokenizer.encode(line)
    v3_tokenized.append(" ".join(map(str, tokens)))

# Tokenize v5 data
print("Tokenizing v5 data...")
v5_tokenized = []
for line in v5_lines:
    tokens = tokenizer.encode(line)
    v5_tokenized.append(" ".join(map(str, tokens)))

# Save tokenized versions
with open("cal_v3_tokenize.txt", "w") as f:
    for tokenized_line in v3_tokenized:
        f.write(tokenized_line + "\n")

with open("cal_v5_tokenize.txt", "w") as f:
    for tokenized_line in v5_tokenized:
        f.write(tokenized_line + "\n")

print("Tokenized data saved to cal_v3_tokenize.txt and cal_v5_tokenize.txt")
print(f"v3: {len(v3_tokenized)} lines tokenized")
print(f"v5: {len(v5_tokenized)} lines tokenized")
