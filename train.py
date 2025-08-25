#!/usr/bin/env python3
import argparse
import json
import os
import time
import traceback
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.optimizers import cosine_decay, linear_schedule, join_schedules
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx.utils import tree_flatten, tree_map

try:
    import wandb
except Exception:
    wandb = None

from embed import extract_embeddings
from cali_data_loader import get_cali_stream
from eval_mteb import evaluate_mteb_tasks
from loss import (
    EmbeddingMimicLoss,
)


def apply_lora_to_model(
    model,
    lora_layers: int | None,
    rank: int,
    scale: float,
    dropout: float,
    lora_keys: set[str],
):
    print("Applying LoRA adapters (MLX style)")
    total_layers = len(model.layers)
    num_target_layers = (
        total_layers
        if (lora_layers is None or lora_layers < 0)
        else min(lora_layers, total_layers)
    )
    print(f"Total layers: {total_layers}")
    print(f"LoRA-applied layers: {num_target_layers}")

    total_params = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters: {total_params:.3f}M")

    model.freeze()

    converted = 0
    target_names = []
    start_idx = total_layers - num_target_layers
    for layer in model.layers[start_idx:]:
        attn = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            full_key = f"self_attn.{proj_name}"
            if full_key in lora_keys and hasattr(attn, proj_name):
                base = getattr(attn, proj_name)
                lora = LoRALinear.from_base(base, r=rank, dropout=dropout, scale=scale)
                setattr(attn, proj_name, lora)
                converted += 1
                target_names.append(full_key)

        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                full_key = f"mlp.{proj_name}"
                if full_key in lora_keys and hasattr(mlp, proj_name):
                    base = getattr(mlp, proj_name)
                    lora = LoRALinear.from_base(
                        base, r=rank, dropout=dropout, scale=scale
                    )
                    setattr(mlp, proj_name, lora)
                    converted += 1
                    target_names.append(full_key)

    print(f"Converted {converted} linear layers to LoRA: {sorted(set(target_names))}")

    trainable_params = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    )
    print(f"Trainable parameters: {trainable_params:.3f}M")
    if trainable_params > 0.0:
        print(f"LoRA applied ({trainable_params/total_params*100:.2f}% trainable)")
    else:
        print("Warning: no trainable parameters detected after LoRA injection")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--model", type=str, default="gemma-3-270m-mlx")
    parser.add_argument("--adapters", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mlx-embedding")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument(
        "--eval-steps", type=int, default=1000, help="Evaluate every N steps using MTEB"
    )
    parser.add_argument(
        "--eval-tasks",
        nargs="*",
        default=None,
        help="MTEB tasks to evaluate (defaults to all nano* tasks)",
    )
    parser.add_argument(
        "--skip-eval-init",
        action="store_true",
        help="Skip initial evaluation at step 0",
    )
    parser.add_argument(
        "--save-steps", type=int, default=1000, help="Save every N steps"
    )
    parser.add_argument(
        "--cali-version", type=str, default="v7", help="Cali dataset version"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )

    args = parser.parse_args()

    # Optimal training parameters based on SOTA research
    max_grad_norm = 1.0  # Optimal: 1.0 (standard for embedding models)
    warmup_ratio = 0.1  # Optimal: 10% warmup (standard practice)

    # Optimal LoRA parameters based on SOTA models (NV-Embed, E5-Mistral, etc.)
    lora_rank = 6  # Optimal: 16 (used by top models)
    lora_alpha = 8.0  # Optimal: 2x rank for best performance
    lora_dropout = 0.2  # Optimal: 0.0 (per Unsloth research - not useful for LoRA)
    lora_layers = -1
    lora_keys = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }

    if args.adapters:
        model, tokenizer = load(args.model, adapter_path=args.adapters)
        try:
            with open(os.path.join(args.adapters, "adapter_config.json"), "r") as f:
                cfg = json.load(f)
            lora_layers = cfg.get("num_layers", lora_layers)
            lp = cfg.get("lora_parameters", {})
            lora_rank = lp.get("rank", lora_rank)
            lora_alpha = lp.get("alpha", lora_alpha)
            lora_dropout = lp.get("dropout", lora_dropout)
            keys = lp.get("keys")
            if isinstance(keys, list) and keys:
                lora_keys = set(keys)
        except Exception as e:
            pass
    else:
        model, tokenizer = load(args.model)
        model = apply_lora_to_model(
            model,
            lora_layers=lora_layers,
            rank=lora_rank,
            scale=lora_alpha,
            dropout=lora_dropout,
            lora_keys=lora_keys,
        )

    # Use embedding mimic loss for training embeddings to match targets
    # Research shows combining L1/L2 with cosine similarity is more effective than cosine alone
    loss_fn = EmbeddingMimicLoss(
        normalize=True,
        alpha=0.6,  # Weight for L2 distance loss (cosine weight automatically 0.4)
        distance_type="l2",  # Use L2 loss for better gradient properties
    )

    # MLX-optimized learning rate and weight decay
    learning_rate = 2e-4  # Optimal: 2e-4 (per Unsloth research for LoRA)
    weight_decay = (
        0.01  # Optimal: 0.01 (per Unsloth research for better regularization)
    )

    # Calculate total steps based on epochs and dataset size
    # Get dataset size for epoch calculation (for logging purposes only)
    sample_stream = get_cali_stream(version=args.cali_version, batch_size=1)
    dataset_size = 0
    try:
        while True:
            next(sample_stream)
            dataset_size += 1
    except StopIteration:
        pass

    print(f"Dataset size: {dataset_size} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Estimated steps per epoch: ~{dataset_size // args.batch_size}")

    # Update learning rate scheduler for estimated total steps
    estimated_steps_per_epoch = dataset_size // args.batch_size
    total_steps = estimated_steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    # Create warmup schedule (linear increase from 0 to learning_rate)
    warmup = linear_schedule(0, learning_rate, warmup_steps)

    # Create cosine decay schedule (from learning_rate to end_lr)
    decay_steps = total_steps - warmup_steps
    cosine = cosine_decay(learning_rate, decay_steps, end=1e-6)
    lr_schedule = join_schedules([warmup, cosine], [warmup_steps])

    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)

    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    trainable_params = sum(
        v.size for _, v in tree_flatten(model.trainable_parameters())
    )

    wb_run = None
    if args.wandb and wandb is not None:
        run_name = (
            args.run_name or f"train-es-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.tags,
            config={
                "epochs": args.epochs,
                "total_steps": total_steps,
                "steps_per_epoch": estimated_steps_per_epoch,
                "dataset_size": dataset_size,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "loss": "embedding_mimic",
                "cali_version": args.cali_version,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_layers": lora_layers,
                "lora_keys": sorted(lora_keys),
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "max_grad_norm": max_grad_norm,
                "warmup_steps": warmup_steps,
                "warmup_ratio": warmup_ratio,
                "evaluation": (
                    "MTEB evaluation enabled"
                    if args.eval_tasks
                    else "MTEB evaluation disabled"
                ),
                "eval_steps": args.eval_steps,
                "eval_tasks": args.eval_tasks if args.eval_tasks else None,
            },
        )

    # Create MLX Data stream that outputs training-ready batches directly
    es_stream = get_cali_stream(
        version=args.cali_version,
        batch_size=args.batch_size,
    )

    # Optional initial evaluation at step 0 (before training)
    start_metrics = None
    if args.eval_tasks and not args.skip_eval_init:
        print("Starting MTEB evaluation...")
        start_metrics = evaluate_mteb_tasks(
            adapter_path=0,  # for step 0
            max_length=args.max_length,
            verbose=True,
            model=model,  # Pass already-loaded model
            tokenizer=tokenizer,  # Pass already-loaded tokenizer
            tasks=args.eval_tasks,
        )

        print(f"Initial MTEB Results:")
        print(f"Average NDCG@5: {start_metrics['avg_ndcg_at_5']:.4f}")
        print(
            f"Tasks evaluated: {start_metrics['valid_tasks']}/{start_metrics['total_tasks']}"
        )

        # Log initial results to wandb if available
        if wb_run is not None:
            for task_name, score in start_metrics["ndcg_at_5_by_task"].items():
                wandb.log({f"eval/ndcg@5/{task_name}": score}, step=0)
    elif args.eval_tasks and args.skip_eval_init:
        print("Skipping initial evaluation (--skip-eval-init specified)")
    else:
        print("Skipping evaluation (no eval-tasks specified)")

    print("\nStarting embedding training...")
    print(
        f"Using learning rate scheduler: cosine decay with {warmup_steps} warmup steps"
    )
    print(
        f"Batch size: {args.batch_size}, gradient accumulation steps: {args.gradient_accumulation_steps}, micro batch size: {args.batch_size // args.gradient_accumulation_steps}"
    )
    print(f"Gradient clipping enabled with max_norm={max_grad_norm}")
    print(
        f"MLX-optimized LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout} (Unsloth-optimized)"
    )
    print(f"Using loss: embedding mimic (cosine similarity)")
    print(f"Learning rate: {learning_rate} (Unsloth-optimized for LoRA)")
    print(f"Weight decay: {weight_decay} (Unsloth-optimized for regularization)")
    print(f"Cali dataset version: {args.cali_version}, epochs: {args.epochs}")

    # Log embedding dimensions
    sample_batch = next(es_stream)
    # Convert numpy arrays to MLX arrays for the model
    sample_batch_mlx = {
        "input_ids": mx.array(sample_batch["tokenized"]),
        "attention_mask": mx.array(sample_batch["attention_mask"]),
        "embedding": mx.array(sample_batch["embedding"]),
    }
    sample_embeddings = extract_embeddings(
        model, sample_batch_mlx["input_ids"], sample_batch_mlx["attention_mask"]
    )
    print(sample_embeddings)
    target_dim = sample_batch_mlx["embedding"].shape[1]
    pred_dim = sample_embeddings.shape[1]
    print(f"Model embedding dimension: {pred_dim}")
    print(f"Target embedding dimension: {target_dim}")
    if target_dim > pred_dim:
        print(
            f"Target embeddings will be truncated from {target_dim} to {pred_dim} dimensions"
        )
    elif target_dim < pred_dim:
        print(
            f"Target embeddings will be padded from {target_dim} to {pred_dim} dimensions"
        )
    else:
        print("Embedding dimensions match perfectly")

    def compute_loss(batch):
        # Extract embeddings from the model for the input tokens
        predicted_embeddings = extract_embeddings(
            model, batch["input_ids"], batch["attention_mask"]
        )

        # Get target embeddings from the batch
        target_embeddings = batch["embedding"]

        # Compute loss between predicted and target embeddings
        # The loss function will automatically handle dimension mismatch
        loss = loss_fn(predicted_embeddings, target_embeddings)
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

    # Helper function to create adapter config
    def create_adapter_config(step=None, best_score=None):
        config = {
            "fine_tune_type": "lora",
            "num_layers": (
                len(model.layers)
                if (lora_layers is None or lora_layers < 0)
                else lora_layers
            ),
            "lora_parameters": {
                "rank": lora_rank,
                "alpha": lora_alpha,
                "scale": lora_alpha,
                "dropout": lora_dropout,
                "keys": sorted(list(lora_keys)),
            },
        }
        if step is not None:
            config["step"] = step
        if best_score is not None:
            config["best_ndcg_at_5"] = best_score
        return config

    for epoch in range(args.epochs):
        print(f"\nStarting epoch {epoch + 1}/{args.epochs}")
        epoch_start_step = epoch * estimated_steps_per_epoch

        # Reset data stream for each epoch
        es_stream = get_cali_stream(
            version=args.cali_version,
            batch_size=args.batch_size,
        )

        step = 0
        for training_batch in es_stream:
            step += 1
            global_step = epoch_start_step + step

            tokens = training_batch["attention_mask"].sum().item()
            batch_tokens = tokens

            t0 = time.perf_counter()
            # get the actual batch size
            batch_size = training_batch["tokenized"].shape[0]
            grad_steps = args.gradient_accumulation_steps
            # get the micro batch size
            micro_batch_size = batch_size // grad_steps

            # Initialize accumulator
            accum_grads = None
            total_loss = 0.0  # Track loss for logging

            for micro_step in range(grad_steps):
                start_idx = micro_step * micro_batch_size
                end_idx = min(start_idx + micro_batch_size, batch_size)

                micro_batch = {
                    "input_ids": mx.array(
                        training_batch["tokenized"][start_idx:end_idx]
                    ),
                    "attention_mask": mx.array(
                        training_batch["attention_mask"][start_idx:end_idx]
                    ),
                    "embedding": mx.array(
                        training_batch["embedding"][start_idx:end_idx]
                    ),
                }
                loss, grads = loss_and_grad_fn(micro_batch)
                total_loss += loss.item()  # Convert to Python float immediately
                if accum_grads is not None:
                    accum_grads = tree_map(mx.add, grads, accum_grads)
                else:
                    accum_grads = grads

                # Clean up micro-batch resources
                del micro_batch
                del grads
                del loss  # Delete loss tensor
                mx.eval(accum_grads)

            # Clean up training batch after all micro-batches processed
            del training_batch
            # Normalize accumulated gradients by total weight
            accum_grads = tree_map(lambda g: g / grad_steps, accum_grads)

            # Add gradient clipping for training stability
            accum_grads, _ = optim.clip_grad_norm(accum_grads, max_grad_norm)

            # Update with accumulated gradients
            optimizer.update(model, accum_grads)
            mx.eval(model.trainable_parameters(), optimizer.state)

            # Clean up accumulated gradients
            del accum_grads

            dt = time.perf_counter() - t0

            avg_loss = total_loss / grad_steps
            token_per_sec = batch_tokens / dt if dt > 0 else 0.0

            # Get current learning rate from scheduler
            current_lr = lr_schedule(global_step)

            print(
                f"Epoch {epoch + 1}/{args.epochs}, Step {global_step}/{total_steps}, Loss: {avg_loss:.4f}, LR: {float(current_lr):.2e}, tokens/sec: {token_per_sec:.0f}"
            )

            if global_step % 10 == 0 and wb_run is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/learning_rate": float(current_lr),
                        "train/tokens_per_sec": token_per_sec,
                        "train/batch_tokens": batch_tokens,
                    },
                    step=global_step,
                )

            if global_step % args.eval_steps == 0 and args.eval_tasks:
                print(f"\nRunning MTEB evaluation at step {global_step}...")
                eval_metrics = evaluate_mteb_tasks(
                    adapter_path=global_step,
                    max_length=args.max_length,
                    verbose=False,
                    model=model,  # Pass already-loaded model
                    tokenizer=tokenizer,  # Pass already-loaded model
                    tasks=args.eval_tasks,
                )

                print(f"Step {global_step} MTEB Results:")
                print(f"Average NDCG@5: {eval_metrics['avg_ndcg_at_5']:.4f}")
                print(
                    f"Tasks evaluated: {eval_metrics['valid_tasks']}/{eval_metrics['total_tasks']}"
                )

                # Log evaluation results to wandb if available
                if wb_run is not None:
                    # Log individual task results - wandb will auto-group them in the same chart
                    for task_name, score in eval_metrics["ndcg_at_5_by_task"].items():
                        wandb.log({f"eval/ndcg@5/{task_name}": score}, step=global_step)

                # Check if this is a new best score and save to best/ directory
                best_score_file = "./adapters/best_score.txt"
                current_best = 0.0
                
                if os.path.exists(best_score_file):
                    with open(best_score_file, "r") as f:
                        current_best = float(f.read().strip())
                
                if eval_metrics['avg_ndcg_at_5'] > current_best:
                    print(f"New best NDCG@5: {eval_metrics['avg_ndcg_at_5']:.4f} (previous: {current_best:.4f})")
                    
                    # Save best score
                    os.makedirs("./adapters/best", exist_ok=True)
                    with open(best_score_file, "w") as f:
                        f.write(f"{eval_metrics['avg_ndcg_at_5']}")
                    
                    # Save best model
                    best_dir = "./adapters/best"
                    model.save_weights(os.path.join(best_dir, "adapters.safetensors"))
                    
                    # Save best model config
                    with open(os.path.join(best_dir, "adapter_config.json"), "w") as f:
                        json.dump(create_adapter_config(step=global_step, best_score=eval_metrics['avg_ndcg_at_5']), f, indent=2)

            if global_step % args.save_steps == 0:
                output_dir = (
                    args.adapters if args.adapters else f"./adapters"
                )
                output_dir = os.path.join(output_dir, f"step_{global_step}")
                os.makedirs(output_dir, exist_ok=True)
                model.save_weights(os.path.join(output_dir, "adapters.safetensors"))

                # Save regular checkpoint config
                with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
                    json.dump(create_adapter_config(step=global_step), f, indent=2)

        print(f"Epoch {epoch + 1} completed after {step} steps")

    # Final evaluation
    if args.eval_tasks:
        print("\nRunning final MTEB evaluation...")
        final_metrics = evaluate_mteb_tasks(
            adapter_path=total_steps,
            max_length=args.max_length,
            verbose=True,
            model=model,  # Pass already-loaded model
            tokenizer=tokenizer,  # Pass already-loaded model
            tasks=args.eval_tasks,
        )

        print(f"\nFinal Training Results:")
        print(f"Final Average NDCG@5: {final_metrics['avg_ndcg_at_5']:.4f}")
        if start_metrics:
            print(
                f"Improvement: {final_metrics['avg_ndcg_at_5'] - start_metrics['avg_ndcg_at_5']:.4f}"
            )

        # Log final results to wandb if available
        if wb_run is not None:
            if start_metrics:
                wandb.log(
                    {
                        "eval/improvement": final_metrics["avg_ndcg_at_5"]
                        - start_metrics["avg_ndcg_at_5"],
                    },
                    step=total_steps,
                    epoch=args.epochs,
                )

            # Log final individual task results - wandb will auto-group them in the same chart
            for task_name, score in final_metrics["ndcg_at_5_by_task"].items():
                wandb.log({f"eval/ndcg@5/{task_name}": score}, step=total_steps)
    else:
        print("\nSkipping final evaluation (no eval-tasks specified)")

    final_output_dir = args.adapters if args.adapters else "./adapters/final"
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_weights(os.path.join(final_output_dir, "adapters.safetensors"))

    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": (
            len(model.layers)
            if (lora_layers is None or lora_layers < 0)
            else lora_layers
        ),
        "lora_parameters": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "keys": sorted(list(lora_keys)),
        },
    }

    with open(os.path.join(final_output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
