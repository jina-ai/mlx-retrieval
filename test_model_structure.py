#!/usr/bin/env python3
"""
Examine the model structure to see where dropout layers are located.
"""

from mlx_lm import load

# Load model
model, tokenizer = load("./gemma-3-270m-mlx", adapter_path="./adapters/best")

print("=== Full Model Structure ===")
print(model)
print("\n" + "=" * 80 + "\n")

print("=== Model Training Mode ===")
print(f"Model training mode: {model.training}")
print(f"Model has training attribute: {hasattr(model, 'training')}")

print("\n=== LoRA Layers with Dropout ===")
# Check if the model has layers attribute
if hasattr(model, "layers"):
    print(f"Number of layers: {len(model.layers)}")

    # Look for LoRA layers in the last few layers (where LoRA is typically applied)
    for i, layer in enumerate(model.layers[-3:], start=len(model.layers) - 3):
        print(f"\nLayer {i}:")
        print(f"  Type: {type(layer)}")

        # Check attention layers
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            print(f"  Self-attention type: {type(attn)}")

            # Check LoRA projections
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    print(f"    {proj_name}: {type(proj)}")
                    if hasattr(proj, "dropout"):
                        print(f"      Dropout: {proj.dropout}")
                    if hasattr(proj, "training"):
                        print(f"      Training mode: {proj.training}")

        # Check MLP layers
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            print(f"  MLP type: {type(mlp)}")

            # Check LoRA projections
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(mlp, proj_name):
                    proj = getattr(mlp, proj_name)
                    print(f"    {proj_name}: {type(proj)}")
                    if hasattr(proj, "dropout"):
                        print(f"      Dropout: {proj.dropout}")
                    if hasattr(proj, "training"):
                        print(f"      Training mode: {proj.training}")

print("\n=== Model Parameters ===")
print(f"Total parameters: {sum(v.size for _, v in model.parameters()) / 1e6:.2f}M")
print(
    f"Trainable parameters: {sum(v.size for _, v in model.trainable_parameters()) / 1e6:.2f}M"
)
