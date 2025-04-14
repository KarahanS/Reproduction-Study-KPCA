import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import matplotlib.pyplot as plt
import os
import sys
import timm
import os
from torchvision import transforms
from glob import glob
from PIL import Image


SUPPORTED_MODELS = [
    # ViT models
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    # DeiT models
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
    "deit_tiny_distilled_patch16_224",
    "deit_small_distilled_patch16_224",
    "deit_base_distilled_patch16_224",
]

image_dir = "C:\\Users\\karab\\Desktop\\imnet_sample"
os.makedirs(image_dir, exist_ok=True)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
image_paths = sorted(glob(f"{image_dir}/*.JPEG"))
images = [transform(Image.open(p).convert("RGB")) for p in image_paths]
input_tensor = torch.stack(images)  # Shape: [B, 3, 224, 224]


for model_name in SUPPORTED_MODELS:
    print(f"\n===== Results for {model_name} =====")

    model = timm.create_model(model_name, pretrained=True)
    attention_outputs = []  # global list to store attention-weighted values

    def attn_detailed_hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attn_out = attn @ v

        attention_outputs.append(
            {
                "x": x.detach(),
                "q": q.detach(),
                "k": k.detach(),
                "v": v.detach(),
                "attn": attn.detach(),
            }
        )

    def remove_all_hooks(model):
        for module in model.modules():
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()
            if hasattr(module, "_forward_pre_hooks"):
                module._forward_pre_hooks.clear()
            if hasattr(module, "_backward_hooks"):
                module._backward_hooks.clear()

    remove_all_hooks(model)

    hook_handles = []
    for blk in model.blocks:
        handle = blk.attn.register_forward_hook(attn_detailed_hook)
        hook_handles.append(handle)

    with torch.no_grad():
        output = model(input_tensor)

    for h in hook_handles:
        h.remove()

    # Initialize lists to store statistics across all images
    all_max_eigs = []
    all_min_eigs = []
    all_mean_eigs = []
    all_median_eigs = []

    # Process each image
    for image_id in range(min(25, input_tensor.shape[0])):
        # Get a sample layer to determine head_dim
        sample_layer = attention_outputs[0]
        sample_q = sample_layer["q"]
        head_dim = sample_q.shape[-1]

        # Store all eigenvalues from all heads and layers for this image
        # We'll organize them by their rank/position
        all_head_layer_eigenvalues = [[] for _ in range(head_dim)]

        # Process each layer and head
        for layer_idx, layer in enumerate(attention_outputs):
            x, q, k, v, attn_matr = (
                layer["x"],
                layer["q"],
                layer["k"],
                layer["v"],
                layer["attn"],
            )
            heads = q.shape[1]  # number of heads

            for h in range(heads):
                k_bh = k[image_id, h]  # [N, head_dim]
                q_bh = q[image_id, h]  # [N, head_dim]
                v_bh = v[image_id, h]  # [N, head_dim]

                head_dim = q.shape[-1]  # head_dim (e.g., 64 for ViT with 16x16 patches)
                num_heads = q.shape[1]  # number of heads
                B = q.shape[0]  # number of images in a batch
                D = q.shape[-1]  # head_dim (e.g., 64 for ViT with 16x16 patches)
                N = q.shape[
                    2
                ]  # number of tokens (e.g., 197 for ViT with 16x16 patches)

                q_bh = (q_bh - q_bh.mean(dim=-1, keepdim=True)) / (
                    q_bh.std(dim=-1, keepdim=True) + 1e-8
                )
                k_bh = (k_bh - k_bh.mean(dim=-1, keepdim=True)) / (
                    k_bh.std(dim=-1, keepdim=True) + 1e-8
                )

                K_raw = torch.exp(
                    k_bh
                    @ k_bh.transpose(-2, -1)
                    / torch.sqrt(torch.tensor(head_dim, dtype=torch.float))
                )
                g_vals = K_raw.sum(dim=-1).unsqueeze(1)  # [N, 1]
                K_phi = K_raw / (g_vals @ g_vals.T + 1e-6)

                one_N = torch.ones(N, N, device=k.device) / N
                K_centered = (
                    K_phi - one_N @ K_phi - K_phi @ one_N + one_N @ K_phi @ one_N
                )

                eigvals, eigvecs = torch.linalg.eigh(K_centered)
                sorted_indices = torch.argsort(eigvals, descending=True)
                eigvals = eigvals[sorted_indices]
                eigvecs = eigvecs[:, sorted_indices]

                # make sure we calculated the eigenvalues, eigenvectors correctly
                assert torch.allclose(
                    eigvecs * eigvals, K_centered @ eigvecs
                ), "Eigenvalue equation not satisfied in general"

                # Get absolute eigenvalues and keep top head_dim
                eigvals_abs = eigvals.abs()[:head_dim]

                # Store each eigenvalue by its rank position
                for i in range(len(eigvals_abs)):
                    if i < head_dim:  # Only store up to head_dim eigenvalues
                        all_head_layer_eigenvalues[i].append(eigvals_abs[i].item())

        # Now we have collected all eigenvalues for all heads and layers for this image
        # Calculate the average value for each eigenvalue position across all heads and layers
        avg_eigenvalues = []
        for i in range(head_dim):
            if all_head_layer_eigenvalues[i]:
                avg_eigenvalues.append(np.mean(all_head_layer_eigenvalues[i]))

        avg_eigenvalues = torch.tensor(avg_eigenvalues)

        image_avg_max = avg_eigenvalues.max().item()
        image_avg_min = avg_eigenvalues.min().item()
        image_avg_mean = avg_eigenvalues.mean().item()
        image_avg_median = torch.median(avg_eigenvalues).item()

        # Store the image's statistics
        all_max_eigs.append(image_avg_max)
        all_min_eigs.append(image_avg_min)
        all_mean_eigs.append(image_avg_mean)
        all_median_eigs.append(image_avg_median)

    # Convert lists to tensors for easier statistical operations
    all_max_eigs = torch.tensor(all_max_eigs)
    all_min_eigs = torch.tensor(all_min_eigs)
    all_mean_eigs = torch.tensor(all_mean_eigs)
    all_median_eigs = torch.tensor(all_median_eigs)

    # Calculate statistics across images
    max_mean = all_max_eigs.mean().item()
    max_std = all_max_eigs.std().item()

    min_mean = all_min_eigs.mean().item()
    min_std = all_min_eigs.std().item()

    mean_mean = all_mean_eigs.mean().item()
    mean_std = all_mean_eigs.std().item()

    median_mean = all_median_eigs.mean().item()
    median_std = all_median_eigs.std().item()

    # Print results for this model
    print(f"Maximum eigenvalue: {max_mean:.6f} ± {max_std:.6f}")
    print(f"Minimum eigenvalue: {min_mean:.6f} ± {min_std:.6f}")
    print(f"Mean eigenvalue: {mean_mean:.6f} ± {mean_std:.6f}")
    print(f"Median eigenvalue: {median_mean:.6f} ± {median_std:.6f}")
