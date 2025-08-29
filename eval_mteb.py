#!/usr/bin/env python3
"""
Evaluate embedding model using MTEB library with MLX model and LoRA adapter.
"""

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
from tqdm import tqdm


class MLXWrapper(Wrapper):
    def __init__(self, model, tokenizer, max_length: int = 2048):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Ensure model is in evaluation mode to disable dropout during inference
        self.model.eval()

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 64,
        **kwargs,
    ) -> np.ndarray:
        embeddings = []
        for i in tqdm(
            range(0, len(sentences), batch_size),
            desc=f"Encoding {task_name} with prompt type {prompt_type.value}",
        ):
            batch_sentences = sentences[i : i + batch_size]
            batch_embeddings = encode_texts(
                self.model,
                self.tokenizer,
                batch_sentences,
                prompt_type.value,
                pooling="eos",
                max_length=self.max_length,
            )
            embeddings.append(np.array(batch_embeddings))
        return np.concatenate(embeddings, axis=0)


def create_mlx_wrapper_from_path(
    model_path: str, adapter_path: str = None, max_length: int = 2048
):
    """Factory function to create MLXWrapper from model path."""
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    return MLXWrapper(model, tokenizer, max_length)


def evaluate_mteb_tasks(
    model_path: str = None,
    adapter_path: str = None,
    max_length: int = 256,
    tasks: list[str] = None,
    verbose: bool = True,
    model=None,  # Accept pre-loaded model
    tokenizer=None,  # Accept pre-loaded tokenizer
) -> dict:
    """
    Evaluate embedding model using MTEB library.

    Args:
        model_path: Path to the base model (only used if model not provided)
        adapter_path: Path to LoRA adapter (only used if model not provided)
        max_length: Maximum sequence length
        tasks: List of task names to evaluate (defaults to all nano* tasks)
        verbose: Whether to print progress
        model: Pre-loaded model (if provided, skips loading)
        tokenizer: Pre-loaded tokenizer (if provided, skips loading)

    Returns:
        Dictionary containing evaluation results
    """
    # Use default nano* tasks if none specified
    if tasks is None:
        tasks = [
            "NanoArguAnaRetrieval",
            "NanoClimateFeverRetrieval",
            "NanoDBPediaRetrieval",
            "NanoFEVERRetrieval",
            "NanoFiQA2018Retrieval",
            "NanoHotpotQARetrieval",
            "NanoMSMARCORetrieval",
            "NanoNFCorpusRetrieval",
            "NanoNQRetrieval",
            "NanoQuoraRetrieval",
            "NanoSCIDOCSRetrieval",
            "NanoSciFactRetrieval",
            "NanoTouche2020Retrieval",
        ]

    if verbose:
        print(f"Evaluating on {len(tasks)} MTEB tasks...")

    # Create ModelMeta with proper metadata (common for both cases)
    if model is not None:
        # For pre-loaded models, use adapter path as name
        adapter_name = f"step-{adapter_path}"
    else:
        # For disk loading, extract adapter name from path
        adapter_name = (
            adapter_path.split("/")[-1]
            if adapter_path and adapter_path.strip()
            else "base"
        )
    model_meta = ModelMeta(
        name=f"hanxiao/gemma-3-270m-mlx-{adapter_name}",
        revision=subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip(),
        release_date="2025-01-01",
        languages=["eng-Latn"],
        license="apache-2.0",
        framework=["PyTorch"],
        max_tokens=max_length,
        embed_dim=640,
        n_parameters=270_000_000,
        memory_usage_mb=1024,
        open_weights=True,
        public_training_code="https://github.com/hanxiao/gemma-training",
        public_training_data="https://github.com/hanxiao/gemma-training/data",
        training_datasets={},
        similarity_fn_name="cosine",
        use_instructions=False,
        reference="https://github.com/hanxiao/gemma-training",
    )

    # Set the appropriate loader based on whether model is pre-loaded
    if model is None:
        # Validate required parameters
        if not model_path:
            raise ValueError("model_path must be provided when model is not provided")

        # Load model using ModelMeta loader
        model_meta.loader = partial(
            create_mlx_wrapper_from_path,
            model_path=model_path,
            adapter_path=adapter_path,
            max_length=max_length,
        )
        # Get the model using the meta loader
        model = model_meta.load_model()
    else:
        # Validate required parameters
        if not tokenizer:
            raise ValueError("tokenizer must be provided when model is provided")

        # Use pre-loaded model with ModelMeta
        model_meta.loader = partial(
            MLXWrapper,
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        # Get the model using the meta loader
        model = model_meta.load_model()

    # Run MTEB evaluation
    mteb_tasks = [mteb.get_task(task) for task in tasks]
    evaluation = mteb.MTEB(tasks=mteb_tasks)
    results = evaluation.run(model, overwrite_results=True, k_values=[1, 5, 10])

    # Extract NDCG@5 results from TaskResult objects
    ndcg_at_5_results = {}
    total_ndcg_at_5 = 0.0
    valid_tasks = 0

    for task_result in results:
        task_name = task_result.task_name

        # Try to get NDCG@5 score from the task results
        try:
            # Look for ndcg_at_5 in the scores - structure: scores[split][0]["ndcg_at_5"]
            ndcg_at_5_score = None
            for split, split_scores in task_result.scores.items():
                if split_scores and len(split_scores) > 0:
                    score_dict = split_scores[0]  # First (and usually only) score dict
                    if "ndcg_at_5" in score_dict:
                        ndcg_at_5_score = score_dict["ndcg_at_5"]
                        break

            if ndcg_at_5_score is not None:
                ndcg_at_5_results[task_name] = ndcg_at_5_score
                total_ndcg_at_5 += ndcg_at_5_score
                valid_tasks += 1
        except Exception as e:
            if verbose:
                print(f"Warning: Could not extract NDCG@5 from {task_name}: {e}")
            continue

    # Calculate average NDCG@5
    avg_ndcg_at_5 = total_ndcg_at_5 / valid_tasks if valid_tasks > 0 else 0.0

    # Create summary results
    summary_results = {
        "ndcg_at_5_by_task": ndcg_at_5_results,
        "avg_ndcg_at_5": avg_ndcg_at_5,
        "total_tasks": len(tasks),
        "valid_tasks": valid_tasks,
        "full_results": results,
    }

    if verbose:
        print(f"MTEB Evaluation Results:")
        print(f"Average NDCG@5: {avg_ndcg_at_5:.4f}")
        print(f"Tasks evaluated: {valid_tasks}/{len(tasks)}")
        print("\nNDCG@5 by task:")
        for task_name, score in ndcg_at_5_results.items():
            print(f"  {task_name}: {score:.4f}")

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./gemma-3-270m-mlx")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    results = evaluate_mteb_tasks(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_length=args.max_length,
        tasks=args.tasks,
    )

    print(f"\nFinal Summary:")
    print(f"Average NDCG@5: {results['avg_ndcg_at_5']:.4f}")
    print(f"Tasks evaluated: {results['valid_tasks']}/{results['total_tasks']}")
