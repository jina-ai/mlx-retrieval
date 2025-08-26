import mteb
from sentence_transformers import SentenceTransformer
import numpy as np
from mlx_lm import load
import os
from tqdm import tqdm

import mteb
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType


import numpy as np

tasks = mteb.get_tasks(tasks=["NanoMSMARCORetrieval"])
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

print(tasks)

# run the evaluation
evaluation = mteb.MTEB(tasks=tasks)
# results = evaluation.run(model)


class MLXWrapper(Wrapper):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = model

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        print(f"Encoding {task_name} with prompt type {prompt_type.value}")
        embeddings = model.encode(sentences, prompt_name=prompt_type.value)
        embeddings = embeddings[:, :640]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings


evaluation.run(MLXWrapper(), overwrite_results=True)
