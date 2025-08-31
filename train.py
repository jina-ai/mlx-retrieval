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
from data_loader import training_batch_stream
from eval_mteb import evaluate_mteb_tasks
from loss import (
    HardNegativeInfoNCELoss,
    InfoNCELoss,
    NTXentLossWithAdvancedMining,
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
    parser.add_argument(
        "--steps", type=int, default=100000, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="InfoNCE temperature (default: 0.1 - optimal for embedding models)",
    )
    parser.add_argument("--model", type=str, default="gemma-3-270m-mlx")
    parser.add_argument("--adapters", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mlx-search")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument(
        "--loss",
        type=str,
        default="hn",
        choices=["hn", "nce", "ntx"],
        help="Loss: hn (HardNeg InfoNCE), nce (InfoNCE), ntx (NT-Xent w/ mining)",
    )
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
        "--save-steps", type=int, default=500, help="Save every N steps"
    )

    args = parser.parse_args()

    # Optimal training parameters based on SOTA research
    prefetch_batches = 32  # Optimal: 8 (good balance of memory vs speed)
    num_threads = 4  # Optimal: 4 (MLX recommendation)
    max_grad_norm = 1.0  # Optimal: 1.0 (standard for embedding models)
    warmup_ratio = 0.1  # Optimal: 10% warmup (standard practice)

    # Optimal LoRA parameters based on SOTA models (NV-Embed, E5-Mistral, etc.)
    lora_rank = 16  # Optimal: 16 (used by top models)
    lora_alpha = 32.0  # Optimal: 2x rank for best performance
    lora_dropout = 0.0  # Optimal: 0.0 (per Unsloth research - not useful for LoRA)
    lora_layers = -1  # Apply to all layers
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
            lora_scale = lp.get("scale", lora_scale)
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

    if args.loss == "hn":
        loss_fn = HardNegativeInfoNCELoss(temperature=args.temperature)
    elif args.loss == "nce":
        loss_fn = InfoNCELoss(temperature=args.temperature)
    else:
        loss_fn = NTXentLossWithAdvancedMining(
            temperature=args.temperature,
            mining_method="perc_pos",
            mining_margin=0.95,
            num_hard_negatives=4,
        )

    # MLX-optimized learning rate and weight decay
    learning_rate = 2e-4  # Optimal: 2e-4 (per Unsloth research for LoRA)
    weight_decay = (
        0.01  # Optimal: 0.01 (per Unsloth research for better regularization)
    )

    # Implement learning rate scheduler with warmup using MLX built-in schedulers
    warmup_steps = max(1, int(args.steps * warmup_ratio))

    # Create warmup schedule (linear increase from 0 to learning_rate)
    warmup = linear_schedule(0, learning_rate, warmup_steps)

    # Create cosine decay schedule (from learning_rate to end_lr)
    decay_steps = args.steps - warmup_steps
    cosine = cosine_decay(learning_rate, decay_steps, end=1e-6)

    # Join warmup and cosine decay schedules
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
                "steps": args.steps,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "mining_method": "perc_pos",
                "mining_margin": 0.95,
                "num_hard_negatives": 4,
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
                "prefetch_batches": prefetch_batches,
                "num_threads": num_threads,
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
    es_stream = training_batch_stream(
        batch_size=args.batch_size,
        prefetch_batches=prefetch_batches,
        num_threads=num_threads,
        max_length=args.max_length,
        tokenizer=tokenizer,
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

    print("\nStarting contrastive training...")
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
    print(f"Using loss: {args.loss}, temperature={args.temperature}")
    if args.loss == "ntx":
        print(f"Hard-Negative Mining: method=perc_pos, margin=0.95, num_negatives=4")
    print(f"Learning rate: {learning_rate} (Unsloth-optimized for LoRA)")
    print(f"Weight decay: {weight_decay} (Unsloth-optimized for regularization)")

    def compute_loss(batch):
        query_embeddings = extract_embeddings(
            model, batch["query_input_ids"], batch["query_attention_mask"]
        )
        doc_embeddings = extract_embeddings(
            model, batch["doc_input_ids"], batch["doc_attention_mask"]
        )

        weights = batch.get("weights", None)
        loss = loss_fn(query_embeddings, doc_embeddings, weights)
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

    for step in range(1, args.steps + 1):
        try:
            # Get next training-ready batch directly from MLX Data stream
            training_batch = next(es_stream)
            q_tokens = training_batch["query_attention_mask"].sum().item()
            d_tokens = training_batch["doc_attention_mask"].sum().item()
            batch_tokens = q_tokens + d_tokens

            t0 = time.perf_counter()
            # get the actual batch size
            batch_size = training_batch["query_input_ids"].shape[0]
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
                    "query_input_ids": mx.array(
                        training_batch["query_input_ids"][start_idx:end_idx]
                    ),
                    "query_attention_mask": mx.array(
                        training_batch["query_attention_mask"][start_idx:end_idx]
                    ),
                    "doc_input_ids": mx.array(
                        training_batch["doc_input_ids"][start_idx:end_idx]
                    ),
                    "doc_attention_mask": mx.array(
                        training_batch["doc_attention_mask"][start_idx:end_idx]
                    ),
                    "weights": mx.array(training_batch["weights"][start_idx:end_idx]),
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
            current_lr = lr_schedule(step)

            print(
                f"Step {step}/{args.steps}, Loss: {avg_loss:.4f}, LR: {float(current_lr):.2e}, tokens/sec: {token_per_sec:.0f}"
            )

            if step % 10 == 0 and wb_run is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/learning_rate": float(current_lr),
                        "train/tokens_per_sec": token_per_sec,
                        "train/batch_tokens": batch_tokens,
                    },
                    step=step,
                )

            if step % args.eval_steps == 0 and args.eval_tasks:
                print(f"\nRunning MTEB evaluation at step {step}...")
                eval_metrics = evaluate_mteb_tasks(
                    adapter_path=step,
                    max_length=args.max_length,
                    verbose=True,
                    model=model,  # Pass already-loaded model
                    tokenizer=tokenizer,  # Pass already-loaded model
                    tasks=args.eval_tasks,
                )

                # Log evaluation results to wandb if available
                if wb_run is not None:
                    # Log individual task results - wandb will auto-group them in the same chart
                    for task_name, score in eval_metrics["ndcg_at_5_by_task"].items():
                        wandb.log({f"eval/ndcg@5/{task_name}": score}, step=step)

            if step % args.save_steps == 0:
                output_dir = (
                    args.adapters if args.adapters else f"./adapters/step_{step}"
                )
                os.makedirs(output_dir, exist_ok=True)
                model.save_weights(os.path.join(output_dir, "adapters.safetensors"))

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

                with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
                    json.dump(adapter_config, f, indent=2)

        except StopIteration:
            print(f"Data stream exhausted at step {step}")
            break
        except Exception as e:
            # print stack trace
            traceback.print_exc()
            print(f"Error in step {step}: {e}")
            continue

    # Final evaluation
    if args.eval_tasks:
        print("\nRunning final MTEB evaluation...")
        final_metrics = evaluate_mteb_tasks(
            adapter_path=args.steps,
            max_length=args.max_length,
            verbose=True,
            model=model,  # Pass already-loaded model
            tokenizer=tokenizer,  # Pass already-loaded tokenizer
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
                    step=args.steps,
                )

            # Log final individual task results - wandb will auto-group them in the same chart
            for task_name, score in final_metrics["ndcg_at_5_by_task"].items():
                wandb.log({f"eval/ndcg@5/{task_name}": score}, step=args.steps)
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
