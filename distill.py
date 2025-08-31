#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.optimizers import cosine_decay, linear_schedule, join_schedules
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.tuner.utils import get_total_parameters

try:
    import wandb
except Exception:
    wandb = None

from embed import extract_eos_embeddings
from cali_data_loader_eos import get_cali_stream
from eval_mteb import evaluate_mteb_tasks


def apply_lora_to_model(
    model,
    lora_layers: int | None,
    rank: int,
    scale: float,
    dropout: float,
    lora_keys: set[str],
):
    total_layers = len(model.layers)
    num_target_layers = (
        total_layers
        if (lora_layers is None or lora_layers < 0)
        else min(lora_layers, total_layers)
    )
    print(f"Total layers: {total_layers}")
    print(f"LoRA-applied layers: {num_target_layers}")
    print(f"Total parameters: {get_total_parameters(model)}")

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

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--model", type=str, default="gemma-3-270m-mlx")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mlx-eos-v5")
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
        "--save-steps", type=int, default=None, help="Save every N steps"
    )
    parser.add_argument(
        "--data-version",
        nargs="*",
        default=["v6", "v7", "v8"],
        help="Calibration data version",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )

    args = parser.parse_args()

    # Optimal training parameters based on SOTA research
    warmup_ratio = 0.1  # Optimal: 10% warmup (standard practice)

    # Optimal LoRA parameters based on SOTA models (NV-Embed, E5-Mistral, etc.)
    lora_rank = 16  # Optimal: 16 (used by top models)
    lora_alpha = 32.0  # Optimal: 2x rank for best performance
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

    if args.adapter:
        with open(os.path.join(args.adapter, "adapter_config.json"), "r") as f:
            cfg = json.load(f)
        lora_layers = cfg.get("num_layers", lora_layers)
        lp = cfg.get("lora_parameters", {})
        lora_rank = lp.get("rank", lora_rank)
        lora_alpha = lp.get("alpha", lora_alpha)
        lora_dropout = lp.get("dropout", lora_dropout)
        keys = lp.get("keys")
        if isinstance(keys, list) and keys:
            lora_keys = set(keys)

    model, tokenizer = load(args.model)
    # remove lm_head if it exists
    if "lm_head" in model.parameters():
        del model.lm_head

    if args.weights:
        model.load_weights(args.weights)
    else:
        if not args.no_lora or args.adapter:
            if not args.no_lora:
                model.freeze()
                # if adapter is provided and no_lora is true, that means we need to fuse
                # the adapter into the model and the model should be trainable overall

            model = apply_lora_to_model(
                model,
                lora_layers=lora_layers,
                rank=lora_rank,
                scale=lora_alpha,
                dropout=lora_dropout,
                lora_keys=lora_keys,
            )
        if args.adapter:
            # load adapter weights and this will make the lora resume-trainable
            model.load_weights(
                os.path.join(args.adapter, "adapters.safetensors"), strict=False
            )
            if args.no_lora:
                # fuse the adapter into the model
                fused_linears = [
                    (n, m.fuse())
                    for n, m in model.named_modules()
                    if hasattr(m, "fuse")
                ]

                if fused_linears:
                    model.update_modules(tree_unflatten(fused_linears))

    model.set_dtype(mx.float32)
    # Ensure model is in training mode for dropout to be active
    model.train()

    # Print all model layers float types
    print("Model layers float types:")

    # Use tree_flatten to recursively access all parameters
    flattened_params = tree_flatten(model.parameters())
    total_params = sum(v.size for _, v in flattened_params)
    trainable_params = sum(
        v.size for _, v in tree_flatten(model.trainable_parameters())
    )
    for name, param in flattened_params:
        if hasattr(param, "dtype"):
            print(f"  {name}: {param.dtype}")
        else:
            print(f"  {name}: {type(param)}")

    # MLX-optimized learning rate and weight decay
    learning_rate = 2e-4  # Optimal: 2e-4 (per Unsloth research for LoRA)
    weight_decay = (
        0.01  # Optimal: 0.01 (per Unsloth research for better regularization)
    )

    # Calculate total tokens from space separated tokens in the file

    num_tokens = 0
    for v in args.data_version:
        with open(f"data/{v}_tokenize.txt", "r") as f:
            num_tokens += sum(len(line.split()) for line in f.readlines())

    print(f"Dataset size: {num_tokens} tokens")

    # Update learning rate scheduler for estimated total steps
    estimated_steps_per_epoch = num_tokens // args.batch_size
    total_steps = estimated_steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    # Create warmup schedule (linear increase from 0 to learning_rate)
    warmup = linear_schedule(0, learning_rate, warmup_steps)

    # Create cosine decay schedule (from learning_rate to end_lr)
    decay_steps = total_steps - warmup_steps
    cosine = cosine_decay(learning_rate, decay_steps, end=1e-6)
    lr_schedule = join_schedules([warmup, cosine], [warmup_steps])

    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)

    print(
        f"Trainable parameters: {trainable_params}/{total_params} = {trainable_params/total_params*100:.2f}%"
    )
    if trainable_params == 0:
        raise ValueError(
            "Warning: no trainable parameters detected after LoRA injection"
        )

    print("\nStarting embedding training...")
    print(
        f"Using learning rate scheduler: cosine decay with {warmup_steps} warmup steps"
    )
    if not args.no_lora:
        print(
            f"MLX-optimized LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout} (Unsloth-optimized)"
        )
    print(f"Learning rate: {learning_rate} (Unsloth-optimized for LoRA)")
    print(f"Weight decay: {weight_decay} (Unsloth-optimized for regularization)")
    print(f"Dataset version: {args.data_version}, epochs: {args.epochs}")

    # Create MLX Data stream that outputs training-ready batches directly
    es_stream = get_cali_stream(version=args.data_version, batch_size=args.batch_size)

    # Log embedding dimensions
    sample_batch = next(es_stream)
    # Convert numpy arrays to MLX arrays for the model
    sample_batch_mlx = {
        "input_ids": mx.array(sample_batch["tokenized"]),
        "eos_pos": mx.array(sample_batch["eos_pos"]),
        "embedding": mx.array(sample_batch["embedding"]),
    }
    sample_embeddings = extract_eos_embeddings(
        model,
        sample_batch_mlx["input_ids"],
        sample_batch_mlx["eos_pos"],
    )

    target_dim = sample_batch_mlx["embedding"].shape[1]
    pred_dim = sample_embeddings.shape[1]
    print(
        f"Model embedding dimension: {pred_dim} \
        dtype: {sample_embeddings.dtype}  \
        sample: {sample_embeddings[0]}"
    )
    print(
        f"Target embedding dimension: {target_dim} \
        dtype: {sample_batch_mlx['embedding'].dtype} \
        sample: {sample_batch_mlx['embedding'][0]}"
    )

    if target_dim != pred_dim:
        raise ValueError(f"Embedding dimensions mismatch {target_dim} != {pred_dim}")

    def compute_loss(batch):
        predicted = extract_eos_embeddings(model, batch["input_ids"], batch["eos_pos"])
        target = batch["embedding"]

        similarity = nn.losses.cosine_similarity_loss(
            predicted, target, reduction="mean"
        )
        cosine_loss = 1 - similarity

        l1_loss = nn.losses.l1_loss(predicted, target, reduction="mean")

        return cosine_loss + 10 * l1_loss

    # Define the state that will be captured by compile
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_with_grad_accum(batch, grad_steps):
        # Initialize accumulator
        accum_grads = None
        total_loss = 0.0

        batch_size = batch["tokenized"].shape[0]
        micro_batch_size = batch_size // grad_steps

        for micro_step in range(grad_steps):
            start_idx = micro_step * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            micro_batch = {
                "input_ids": batch["tokenized"][start_idx:end_idx],
                "eos_pos": batch["eos_pos"][start_idx:end_idx],
                "embedding": batch["embedding"][start_idx:end_idx],
            }

            loss, grads = loss_and_grad_fn(micro_batch)
            total_loss += loss

            if accum_grads is not None:
                accum_grads = tree_map(mx.add, grads, accum_grads)
            else:
                accum_grads = grads

        if grad_steps > 1:
            # Normalize accumulated gradients
            accum_grads = tree_map(lambda g: g / grad_steps, accum_grads)

        # Gradient clipping
        # accum_grads, _ = optim.clip_grad_norm(accum_grads, max_grad_norm)

        # Update with accumulated gradients
        optimizer.update(model, accum_grads)

        return total_loss / grad_steps

    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)

    # Helper function to create adapter config
    def create_adapter_config(step=None, best_score=None):
        if not args.no_lora:
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
        else:
            config = {
                "fine_tune_type": "full",
            }
        if step is not None:
            config["step"] = step
        if best_score is not None:
            config["best_ndcg_at_5"] = best_score
        return config

    wb_run = None

    step = 0
    accum_loss = 0.0

    for epoch in range(args.epochs):
        if epoch == 0:
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

                model.train()

            if wb_run is None and args.wandb and wandb is not None:
                run_name = (
                    args.run_name or f"emb-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
                        "dataset_num_tokens": num_tokens,
                        "max_length": args.max_length,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "data_version": args.data_version,
                        "lora_rank": lora_rank,
                        "lora_alpha": lora_alpha,
                        "lora_dropout": lora_dropout,
                        "lora_layers": lora_layers,
                        "lora_keys": sorted(lora_keys),
                        "total_params": int(total_params),
                        "trainable_params": int(trainable_params),
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
            # Log initial results to wandb if available
            if wb_run is not None and args.eval_tasks and not args.skip_eval_init:
                for task_name, score in start_metrics["ndcg_at_5_by_task"].items():
                    wandb.log({f"eval/ndcg@5/{task_name}": score}, step=0)

        print(f"\nStarting epoch {epoch + 1}/{args.epochs}")

        # Reset data stream for each epoch
        es_stream = get_cali_stream(
            version=args.data_version, batch_size=args.batch_size
        )

        for training_batch in es_stream:
            step += 1

            tokens = training_batch["eos_pos"].sum().item()
            batch_tokens = tokens

            t0 = time.perf_counter()

            # Convert batch to MLX arrays
            mlx_batch = {
                "tokenized": mx.array(training_batch["tokenized"]),
                "eos_pos": mx.array(training_batch["eos_pos"]),
                "embedding": mx.array(training_batch["embedding"]),
            }

            # Execute compiled training step
            avg_loss = step_with_grad_accum(mlx_batch, args.gradient_accumulation_steps)

            # Evaluate state to ensure updates are applied
            mx.eval(state)

            # Convert MLX array to Python scalar for logging
            avg_loss_scalar = avg_loss.item()
            accum_loss += avg_loss_scalar

            dt = time.perf_counter() - t0
            token_per_sec = batch_tokens / dt if dt > 0 else 0.0

            # Get current learning rate from scheduler
            current_lr = lr_schedule(step)

            print(
                f"Epoch {epoch + 1}/{args.epochs}, Step {step}/{total_steps}, Loss: {avg_loss_scalar:.4f}, LR: {float(current_lr):.2e}, tokens/sec: {token_per_sec:.0f}"
            )

            if step % 10 == 0 and wb_run is not None:
                wandb.log(
                    {
                        "train/loss": accum_loss / 10,
                        "train/learning_rate": float(current_lr),
                        "train/tokens_per_sec": token_per_sec,
                        "train/batch_tokens": batch_tokens,
                    },
                    step=step,
                )
                accum_loss = 0.0

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

                # Restore training mode after evaluation
                model.train()

                # Log evaluation results to wandb if available
                if wb_run is not None:
                    # Log individual task results - wandb will auto-group them in the same chart
                    for task_name, score in eval_metrics["ndcg_at_5_by_task"].items():
                        wandb.log({f"eval/ndcg@5/{task_name}": score}, step=step)

                # Check if this is a new best score and save to best/ directory
                best_score_file = "./adapters/best_score.txt"
                current_best = 0.0

                if os.path.exists(best_score_file):
                    with open(best_score_file, "r") as f:
                        current_best = float(f.read().strip())

                if eval_metrics["avg_ndcg_at_5"] > current_best:
                    print(
                        f"New best NDCG@5: {eval_metrics['avg_ndcg_at_5']:.4f} (previous: {current_best:.4f})"
                    )

                    # Save best score
                    os.makedirs("./adapters/best", exist_ok=True)
                    with open(best_score_file, "w") as f:
                        f.write(f"{eval_metrics['avg_ndcg_at_5']}")

                    # Save best model
                    best_dir = "./adapters/best"
                    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
                    mx.save_safetensors(
                        str(os.path.join(best_dir, "adapters.safetensors")),
                        adapter_weights,
                    )
                    print(f"Saved best model to {best_dir}")

                    # Save best model config
                    with open(os.path.join(best_dir, "adapter_config.json"), "w") as f:
                        json.dump(
                            create_adapter_config(
                                step=step,
                                best_score=eval_metrics["avg_ndcg_at_5"],
                            ),
                            f,
                            indent=2,
                        )

            if args.save_steps and step % args.save_steps == 0:
                output_dir = args.adapter if args.adapter else f"./adapters"
                output_dir = os.path.join(output_dir, f"step_{step}")
                os.makedirs(output_dir, exist_ok=True)
                adapter_weights = dict(tree_flatten(model.trainable_parameters()))
                mx.save_safetensors(
                    str(os.path.join(output_dir, "adapters.safetensors")),
                    adapter_weights,
                )

                # Save regular checkpoint config
                with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
                    json.dump(create_adapter_config(step=step), f, indent=2)
                print(f"Saved checkpoint to {output_dir}")

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

        # Log final results to wandb if available
        if wb_run is not None:
            # Log final individual task results - wandb will auto-group them in the same chart
            for task_name, score in final_metrics["ndcg_at_5_by_task"].items():
                wandb.log({f"eval/ndcg@5/{task_name}": score}, step=total_steps)
    else:
        print("\nSkipping final evaluation (no eval-tasks specified)")

    final_output_dir = args.adapter if args.adapter else "./adapters/final"
    os.makedirs(final_output_dir, exist_ok=True)
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(
        str(os.path.join(final_output_dir, "adapters.safetensors")), adapter_weights
    )

    with open(os.path.join(final_output_dir, "adapter_config.json"), "w") as f:
        json.dump(create_adapter_config(step=total_steps), f, indent=2)

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
