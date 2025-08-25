import mteb
from sentence_transformers import SentenceTransformer
import numpy as np
from mlx_lm import load
import os
from tqdm import tqdm

import mteb
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType

from mteb.model_meta import ModelMeta
import numpy as np
from embed import encode_texts
from mlx_lm import load
import argparse
from functools import partial
import subprocess


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
        return embeddings[:,:640]

evaluation.run(MLXWrapper(), overwrite_results=True)