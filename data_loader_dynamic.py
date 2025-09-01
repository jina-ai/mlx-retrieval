#!/usr/bin/env python3

import mlx.data as dx
import numpy as np
import random
import os

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EOS_TOKEN_ID = 1

all_lines = []
for version in ["v6", "v7", "v8"]:
    with open(f"data/{version}.txt", "r") as f:
        all_lines.extend([line.strip() for line in f.readlines()])
random.shuffle(all_lines)



def sample_with_embedding(qwen_model, tokenizer, batch_size=64):
    batch_samples = []
    batch_texts = []
    
    for s in sample_generator(tokenizer):
        batch_samples.append(s)
        batch_texts.append(s["text"])
        
        if len(batch_samples) == batch_size:
            # Encode all texts at once
            embeddings = qwen_model.encode(batch_texts)[:, :640]
            
            # Yield each sample with its embedding
            for i, sample in enumerate(batch_samples):
                yield {
                    "eos_pos": sample["eos_pos"],
                    "tokenized": sample["tokenized"],
                    "embedding": embeddings[i]
                }
            
            # Reset batch
            batch_samples = []
            batch_texts = []
    
    # Handle remaining samples
    if batch_samples:
        embeddings = qwen_model.encode(batch_texts)[:, :640]
        for i, sample in enumerate(batch_samples):
            yield {
                "eos_pos": sample["eos_pos"],
                "tokenized": sample["tokenized"],
                "embedding": embeddings[i]
            }

def sample_generator(tokenizer):
    for line in all_lines:
        result = line
        case = random.randint(1, 8)
        if case == 1:
            result = line
        elif case == 2:
            start = random.randint(0, len(line) - 1)
            end = random.randint(start + 1, len(line))
            result = line[start:end]
        elif case == 3:
            repeat_count = random.randint(1, 3)
            result = line * repeat_count
        elif case == 4:
            other_line = random.choice(all_lines)
            if random.choice([True, False]):
                result = f"{other_line} {line}"
            else:
                result = f"{line} {other_line}"
        elif case == 5:
            words = line.split()
            random.shuffle(words)
            result = " ".join(words)
        elif case == 6:
            chars = list(line)
            random.shuffle(chars)
            result = "".join(chars)
        elif case == 7:
            mid = len(line) // 2
            result = line[:mid] + line[mid:][::-1]
        elif case == 8:
            words = line.split()
            random_word = random.choice(words)
            result = line.replace(random_word, random_word[::-1])         
        
        role = random.choice(["query", "document"])
        if role == "query":
            tokenized = tokenizer.encode(f"{result}<unused0><eos>")
        else:
            tokenized = tokenizer.encode(f"{result}<unused1><eos>")

        eos_pos = len(tokenized) - 1 - tokenized[::-1].index(EOS_TOKEN_ID)

        yield {
            "text": result,
            "role": role,
            "eos_pos": eos_pos,
            "tokenized": tokenized
        }

def get_cali_stream(qwen_model, tokenizer, batch_size=64):
    stream = dx.stream_python_iterable(lambda: sample_with_embedding(qwen_model, tokenizer))
    stream = stream.batch(batch_size).prefetch(8, 4)
    return stream