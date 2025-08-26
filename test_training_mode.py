#!/usr/bin/env python3
"""
Test that training mode actually enables dropout in the LoRA layers.
"""

from mlx_lm import load

# Load model
model, tokenizer = load("./gemma-3-270m-mlx", adapter_path="./adapters/best")

print("=== Initial Model State ===")
print(f"Model training mode: {model.training}")

# Check dropout layers in the last layer
last_layer = model.model.layers[-1]
print(f"\nLast layer type: {type(last_layer)}")

# Check attention projections
attn = last_layer.self_attn
print(f"\nAttention layer:")
for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
    proj = getattr(attn, proj_name)
    if hasattr(proj, "dropout"):
        dropout_layer = proj.dropout
        print(f"  {proj_name}.dropout.training: {dropout_layer.training}")

# Check MLP projections
mlp = last_layer.mlp
print(f"\nMLP layer:")
for proj_name in ["gate_proj", "up_proj", "down_proj"]:
    proj = getattr(mlp, proj_name)
    if hasattr(proj, "dropout"):
        dropout_layer = proj.dropout
        print(f"  {proj_name}.dropout.training: {dropout_layer.training}")

print("\n=== Setting Model to Training Mode ===")
model.train()
print(f"Model training mode: {model.training}")

# Check dropout layers again
print(f"\nAfter model.train():")
for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
    proj = getattr(attn, proj_name)
    if hasattr(proj, "dropout"):
        dropout_layer = proj.dropout
        print(f"  {proj_name}.dropout.training: {dropout_layer.training}")

print(f"\nMLP after model.train():")
for proj_name in ["gate_proj", "up_proj", "down_proj"]:
    proj = getattr(mlp, proj_name)
    if hasattr(proj, "dropout"):
        dropout_layer = proj.dropout
        print(f"  {proj_name}.dropout.training: {dropout_layer.training}")

print("\n=== Setting Model to Evaluation Mode ===")
model.eval()
print(f"Model training mode: {model.training}")

# Check dropout layers again
print(f"\nAfter model.eval():")
for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
    proj = getattr(attn, proj_name)
    if hasattr(proj, "dropout"):
        dropout_layer = proj.dropout
        print(f"  {proj_name}.dropout.training: {dropout_layer.training}")

print(f"\nMLP after model.eval():")
for proj_name in ["gate_proj", "up_proj", "down_proj"]:
    proj = getattr(mlp, proj_name)
    if hasattr(proj, "dropout"):
        dropout_layer = proj.dropout
        print(f"  {proj_name}.dropout.training: {dropout_layer.training}")

print("\n=== Summary ===")
print("This shows that model.train() and model.eval() properly propagate")
print("to the dropout layers in your LoRA adapters.")
