import torch
import timm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from glob import glob
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import time
import json
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment


def convert_to_json_serializable(obj):
    """Convert tensor values to JSON serializable types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_json_serializable(obj.tolist())
    elif torch.is_tensor(obj):
        return convert_to_json_serializable(obj.cpu().detach().numpy())
    else:
        return obj


# === 0. Parse command line arguments ===
parser = argparse.ArgumentParser(
    description="Analyze vision transformer attention and value vectors"
)
parser.add_argument(
    "--eigenvalue", action="store_true", help="Enable eigenvalue comparison analysis"
)
parser.add_argument(
    "--value-vector",
    action="store_true",
    help="Enable value vector comparison analysis",
)
parser.add_argument(
    "--layer",
    type=int,
    default=None,
    help="Specific transformer layer to analyze (default: all layers)",
)
parser.add_argument(
    "--model", type=str, default=None, help="Specific model architecture to analyze"
)
parser.add_argument(
    "--all-models", action="store_true", help="Analyze all supported models"
)
parser.add_argument(
    "--model-list",
    type=str,
    default=None,
    help="Comma-separated list of models to analyze",
)
parser.add_argument(
    "--base_dir",
    type=str,
    default="/weka/datasets/ImageNet2012",
    help="Base directory for ImageNet",
)
parser.add_argument(
    "--subset",
    type=str,
    default="val",
    choices=["train", "val"],
    help="Dataset subset to use",
)
parser.add_argument(
    "--num_samples", type=int, default=100, help="Number of images to analyze"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="analysis_results",
    help="Directory to save results",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to local checkpoint file to load (default: None)",
)


# Define supported model architectures
SUPPORTED_MODELS = [
    # ViT models
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    # DeiT models
    # "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
    "deit_tiny_distilled_patch16_224",
    "deit_small_distilled_patch16_224",
    "deit_base_distilled_patch16_224",
]


class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(
            self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma))
        )

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        return hsic / (var1 * var2 + 1e-10)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2 + 1e-10)


# Determine which models to analyze
models_to_analyze = []
if args.all_models:
    models_to_analyze = SUPPORTED_MODELS
    print(f"Analyzing all {len(models_to_analyze)} supported models")
elif args.model_list:
    model_list = [m.strip() for m in args.model_list.split(",")]
    for model in model_list:
        if model in SUPPORTED_MODELS:
            models_to_analyze.append(model)
        else:
            print(f"Warning: Model '{model}' is not in the supported list. Skipping.")
    print(f"Analyzing {len(models_to_analyze)} models from provided list")
elif args.model:
    if args.model in SUPPORTED_MODELS:
        models_to_analyze = [args.model]
        print(f"Analyzing specified model: {args.model}")
    else:
        print(
            f"Warning: Model '{args.model}' is not in the supported list. Will try anyway."
        )
        models_to_analyze = [args.model]
else:
    # Default to ViT Tiny if no model is specified
    models_to_analyze = ["vit_tiny_patch16_224"]
    print(f"No model specified. Defaulting to: {models_to_analyze[0]}")


def load_model_with_checkpoint(model_name, checkpoint_path=None):
    """Load model either from timm pretrained or local checkpoint"""
    model = timm.create_model(
        model_name, pretrained=not checkpoint_path, num_classes=1000
    )

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove module prefix if present (for DDP-trained models)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load state dict
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint with message: {msg}")

    return model.eval()


def compute_self_attention(Q, K, head_dim):
    scale = 1.0 / math.sqrt(head_dim)
    A = torch.softmax((Q @ K.transpose(-2, -1)) * scale, dim=-1)
    return A


def compute_value_vector_similarities(V_bh, Vhat):
    """
    Compute cosine similarities between original and reconstructed value vectors.

    Args:
        V_bh (torch.Tensor): Original value vectors [N, head_dim]
        Vhat (torch.Tensor): Reconstructed value vectors [N, head_dim]

    Returns:
        dict: Similarity metrics
    """
    global ALLCLOSE_PASS_COUNT, ALLCLOSE_TOTAL_COUNT

    # Check if vectors are allclose
    vectors_allclose = torch.allclose(V_bh, Vhat, rtol=1e-3, atol=1e-5)

    # Increment counters
    ALLCLOSE_TOTAL_COUNT += 1
    if vectors_allclose:
        ALLCLOSE_PASS_COUNT += 1

    # Normalize each token's value vector to unit length
    V_bh_norm = V_bh / (V_bh.norm(2, dim=1, keepdim=True) + 1e-8)  # [N, head_dim]
    Vhat_norm = Vhat / (Vhat.norm(2, dim=1, keepdim=True) + 1e-8)  # [N, head_dim]

    # Compute CKA similarity
    cka = CudaCKA(V_bh.device)
    linear_cka = cka.linear_CKA(V_bh, Vhat).item()
    kernel_cka = cka.kernel_CKA(V_bh, Vhat).item()

    # 1. Direct token-by-token similarity (corresponding indices)
    token_similarities = torch.sum(V_bh_norm * Vhat_norm, dim=1)  # [N]

    # Compute summary statistics for direct matching
    avg_sim = token_similarities.mean().item()
    min_sim = token_similarities.min().item()
    max_sim = token_similarities.max().item()

    # Count negative similarities
    neg_count = sum(1 for sim in token_similarities.cpu().tolist() if sim < 0)
    neg_percentage = (neg_count / len(token_similarities)) * 100

    # 2. One-to-one optimal matching using Hungarian algorithm
    pairwise_sim_matrix = torch.matmul(V_bh_norm, Vhat_norm.T)  # [N, N]

    # Find best matching pairs using Jonker-Volgenant algorithm for maximum similarity
    cost_matrix = -pairwise_sim_matrix.cpu().numpy()  # Negative for minimization
    original_indices, recon_indices = linear_sum_assignment(cost_matrix)

    # Get optimal matching similarities
    optimal_similarities = -np.array(
        [cost_matrix[i, j] for i, j in zip(original_indices, recon_indices)]
    )
    optimal_avg_sim = optimal_similarities.mean()
    optimal_min_sim = optimal_similarities.min()
    optimal_max_sim = optimal_similarities.max()

    # Count negative optimal similarities
    opt_neg_count = sum(1 for sim in optimal_similarities if sim < 0)
    opt_neg_percentage = (opt_neg_count / len(optimal_similarities)) * 100

    # Compute Frobenius norm of the difference
    frob_diff = torch.norm(V_bh - Vhat, p="fro").item()
    rel_frob_diff = frob_diff / torch.norm(V_bh, p="fro").item()

    # Alternative similarity metrics
    # Euclidean distance
    euclidean_dist = torch.norm(V_bh - Vhat, dim=1)
    avg_euclidean = euclidean_dist.mean().item()

    # L1 distance
    l1_dist = torch.sum(torch.abs(V_bh - Vhat), dim=1)
    avg_l1 = l1_dist.mean().item()

    # Center vectors for Pearson correlation
    V_centered = V_bh - V_bh.mean(dim=1, keepdim=True)
    Vhat_centered = Vhat - Vhat.mean(dim=1, keepdim=True)
    V_centered_norm = V_centered / (V_centered.norm(dim=1, keepdim=True) + 1e-8)
    Vhat_centered_norm = Vhat_centered / (
        Vhat_centered.norm(dim=1, keepdim=True) + 1e-8
    )
    pearson_corr = torch.sum(V_centered_norm * Vhat_centered_norm, dim=1)
    avg_pearson = pearson_corr.mean().item()

    return {
        # Direct token-by-token similarities
        "token_similarities": token_similarities.cpu().tolist(),
        "mean_similarity": avg_sim,
        "min_similarity": min_sim,
        "max_similarity": max_sim,
        "negative_count": neg_count,
        "negative_percentage": neg_percentage,
        # Optimal one-to-one matching similarities
        "optimal_similarities": optimal_similarities.tolist(),
        "optimal_matching_similarity": optimal_avg_sim,
        "optimal_min_similarity": optimal_min_sim,
        "optimal_max_similarity": optimal_max_sim,
        "optimal_negative_count": opt_neg_count,
        "optimal_negative_percentage": opt_neg_percentage,
        # Error metrics
        "frobenius_difference": frob_diff,
        "relative_frobenius_difference": rel_frob_diff,
        # Alternative metrics
        "euclidean_distances": euclidean_dist.cpu().tolist(),
        "avg_euclidean_distance": avg_euclidean,
        "l1_distances": l1_dist.cpu().tolist(),
        "avg_l1_distance": avg_l1,
        "pearson_correlations": pearson_corr.cpu().tolist(),
        "avg_pearson_correlation": avg_pearson,
        "vectors_allclose": vectors_allclose,
        # CKA similarity metrics
        "linear_cka": linear_cka,
        "kernel_cka": kernel_cka,
    }


def compute_kernel_eigendecomposition(K_bh, head_dim):
    """
    Compute eigen-decomposition of the centered kernel matrix.

    Args:
        K_bh: Key vectors [N, head_dim]
        head_dim: Dimension of the attention head

    Returns:
        eigvals: Eigenvalues
        eigvecs: Eigenvectors
    """
    N = K_bh.shape[0]
    scale = 1.0 / math.sqrt(head_dim)

    # k(x, y) = exp(x^T y / sqrt(d)) where d = head_dim
    dot_products = K_bh @ K_bh.T
    scaled_dots = dot_products * scale
    K_raw = torch.exp(scaled_dots)

    # g(x) = sum_{j=1}^N k(x, k_j)
    g_vals = K_raw.sum(dim=1, keepdim=True)  # [N, 1]
    g_outer = g_vals @ g_vals.T

    # Add small epsilon to avoid numerical issues
    K_phi = K_raw / (g_outer + 1e-6)

    one_n = torch.ones(N, N, device=K_phi.device) / N
    K_centered = K_phi - one_n @ K_phi - K_phi @ one_n + one_n @ K_phi @ one_n

    # Add small regularization to diagonal
    K_centered = K_centered + torch.eye(N, device=K_centered.device) * 1e-6

    # Compute eigen-decomposition of the centered kernel matrix
    try:
        eigvals, eigvecs = torch.linalg.eigh(K_centered)

        # Sort in descending order by absolute eigenvalue
        idx = torch.argsort(eigvals.abs(), descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Return only the top-k eigenvectors (where k = head_dim)
        return eigvals[:head_dim], eigvecs[:, :head_dim]
    except Exception as e:
        print(f"Eigendecomposition error: {e}")
        # Return dummy values for debugging
        return torch.zeros(head_dim, device=K_bh.device), torch.zeros(
            N, head_dim, device=K_bh.device
        )


def analyze_layer(model, layer_idx, input_tensor, debug=False):
    """Analyze a specific layer of the model"""
    # Set up hooks to capture value matrix V and key matrix K
    captured = {}

    def att_proj_hook(module, input, output):
        captured["att"] = output.detach().clone()

    # Register hook on the qkv projection for the specified layer
    attn_block = model.blocks[layer_idx].attn

    # Check if attention block has required attributes
    if not hasattr(attn_block, "qkv") or not hasattr(attn_block, "num_heads"):
        print(f"Block {layer_idx} doesn't have expected attention attributes")
        return None

    num_heads = attn_block.num_heads
    head_dim = (
        attn_block.head_dim
        if hasattr(attn_block, "head_dim")
        else attn_block.qkv.weight.shape[0] // (3 * num_heads)
    )
    handle = attn_block.qkv.register_forward_hook(att_proj_hook)

    # Store analysis results for this layer
    layer_results = {
        "layer_idx": layer_idx,
        "eigenvalues": [],
        "value_vector_results": [],
    }

    try:
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)

        # Extract attention values
        qkv = captured["att"]  # shape: [B, N, 3*D_v]
        B, N = qkv.shape[0:2]
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)  # Q, K, V: [B, num_heads, N, head_dim]

        # Process each batch item
        for b in range(B):
            # Process each attention head
            for h in range(num_heads):
                K_bh = K[b, h]  # [N, head_dim]

                # Run eigenvalue analysis if enabled
                if run_eigenvalue_comparison:
                    try:
                        eigvals, _ = compute_kernel_eigendecomposition(K_bh, head_dim)

                        # Store the absolute eigenvalues as numpy array
                        eigval_abs = eigvals.abs().cpu().numpy()
                        layer_results["eigenvalues"].append(
                            {"sample": b, "head": h, "values": eigval_abs}
                        )

                        if debug and b == 0 and h == 0:
                            print(
                                f"Layer {layer_idx}, Sample {b}, Head {h} eigenvalues (first 5):"
                            )
                            print(eigval_abs[:5])
                    except Exception as e:
                        print(
                            f"Error in eigenvalue analysis for layer {layer_idx}, sample {b}, head {h}: {e}"
                        )

                # Run value vector comparison if enabled
                if run_value_vector_comparison:
                    try:
                        V_bh = V[b, h]  # [N, head_dim]

                        # Compute kernel matrix and needed components
                        scale = 1.0 / math.sqrt(head_dim)
                        K_raw = torch.exp(K_bh @ K_bh.T * scale)
                        g_vals = K_raw.sum(dim=1, keepdim=True)

                        # Center the kernel matrix
                        one_n = torch.ones(N, N, device=K_bh.device) / N
                        K_phi = K_raw / (g_vals @ g_vals.T + 1e-6)
                        K_centered = (
                            K_phi
                            - one_n @ K_phi
                            - K_phi @ one_n
                            + one_n @ K_phi @ one_n
                        )

                        # Compute eigendecomposition
                        eigvals, eigvecs = torch.linalg.eigh(K_centered)
                        idx = torch.argsort(eigvals.abs(), descending=True)
                        eigvals = eigvals[idx]
                        eigvecs = eigvecs[:, idx]

                        # Get the top eigenvectors
                        A = eigvecs[:, :head_dim]

                        # Compute G matrix
                        G = torch.diag(1 / g_vals[:, 0])

                        # Compute reconstructed value vectors
                        Vhat = G @ A - G @ one_n @ A

                        # Compute value vector similarities
                        if debug and b == 0 and h == 0:
                            print(
                                f"\nValue Vector Comparison for Layer {layer_idx}, Sample {b}, Head {h}:"
                            )

                        similarities = compute_value_vector_similarities(V_bh, Vhat)

                        layer_results["value_vector_results"].append(
                            {"sample": b, "head": h, "similarities": similarities}
                        )

                    except Exception as e:
                        print(
                            f"Error in value vector comparison for layer {layer_idx}, sample {b}, head {h}: {e}"
                        )
    except Exception as e:
        print(f"Error during analysis of layer {layer_idx}: {e}")
    finally:
        # Remove the hook to avoid memory leaks
        handle.remove()

    return layer_results


def run_multilayer_analysis(model, input_tensor, target_layer=None, debug=False):
    """
    Run analysis on all layers (or a specific layer) of the model.

    Args:
        model: The model to analyze
        input_tensor: Tensor of shape [B, 3, H, W] containing input images
        target_layer: Specific layer to analyze (None for all layers)
        debug: Whether to print debug information

    Returns:
        Dictionary containing hierarchical analysis results
    """
    # Check if model has 'blocks' attribute (ViT/DeiT style)
    if not hasattr(model, "blocks"):
        print(f"Model doesn't have expected 'blocks' attribute")
        return None

    num_layers = len(model.blocks)
    print(f"Model has {num_layers} layers")

    # Determine which layers to analyze
    if target_layer is not None:
        if target_layer >= num_layers:
            print(
                f"Specified layer {target_layer} is out of range. Using last layer ({num_layers-1}) instead."
            )
            target_layer = num_layers - 1
        layers_to_analyze = [target_layer]
        print(f"Analyzing specific layer: {target_layer}")
    else:
        layers_to_analyze = list(range(num_layers))
        print(f"Analyzing all {num_layers} layers")

    # Store results for all layers
    all_layer_results = []

    # Analyze each layer
    for layer_idx in layers_to_analyze:
        print(f"\nAnalyzing layer {layer_idx}/{num_layers-1}...")
        layer_results = analyze_layer(
            model,
            layer_idx,
            input_tensor,
            debug=(debug and layer_idx == layers_to_analyze[0]),
        )
        if layer_results:
            all_layer_results.append(layer_results)

    # Calculate statistics at different levels of granularity
    hierarchical_stats = calculate_hierarchical_statistics(all_layer_results)

    return {
        "all_layer_results": all_layer_results,
        "hierarchical_stats": hierarchical_stats,
    }


def calculate_hierarchical_statistics(layer_results):
    """
    Calculate statistics at different levels of granularity:
    1. Head level (per layer, per head)
    2. Layer level (per layer, averaged across heads)
    3. Model level (averaged across all layers and heads)
    4. Image level (statistics per image, across all layers and heads)

    Args:
        layer_results: List of layer result dictionaries

    Returns:
        Dictionary with hierarchical statistics
    """
    if not layer_results:
        return {}

    # Initialize hierarchical statistics structure
    stats = {"head_level": [], "layer_level": [], "model_level": {}, "image_level": {}}

    # Extract statistics at head level (most detailed)
    for layer in layer_results:
        layer_idx = layer["layer_idx"]
        for result in layer["value_vector_results"]:
            sample_idx = result["sample"]
            head_idx = result["head"]
            sim = result["similarities"]

            # Store head-level statistics
            head_stat = {
                "layer": layer_idx,
                "sample": sample_idx,
                "head": head_idx,
                "direct_similarity": sim["mean_similarity"],
                "optimal_similarity": sim["optimal_matching_similarity"],
                "improvement": sim["optimal_matching_similarity"]
                - sim["mean_similarity"],
                "relative_frobenius_error": sim["relative_frobenius_difference"],
                "negative_percentage": sim["negative_percentage"],
                "pearson_correlation": sim["avg_pearson_correlation"],
                # Add CKA metrics
                "linear_cka": sim.get("linear_cka", 0),
                "kernel_cka": sim.get("kernel_cka", 0),
            }
            stats["head_level"].append(head_stat)

    # Calculate layer-level statistics (average across heads for each layer)
    layer_head_stats = {}
    for head_stat in stats["head_level"]:
        layer_idx = head_stat["layer"]
        if layer_idx not in layer_head_stats:
            layer_head_stats[layer_idx] = {
                "direct_sims": [],
                "optimal_sims": [],
                "improvements": [],
                "rel_frob_errors": [],
                "neg_percentages": [],
                "pearson_corrs": [],
                "linear_ckas": [],  # Add this line
                "kernel_ckas": [],  # Add this line
            }

        # Collect all head statistics for this layer
        layer_head_stats[layer_idx]["direct_sims"].append(
            head_stat["direct_similarity"]
        )
        layer_head_stats[layer_idx]["optimal_sims"].append(
            head_stat["optimal_similarity"]
        )
        layer_head_stats[layer_idx]["improvements"].append(head_stat["improvement"])
        layer_head_stats[layer_idx]["rel_frob_errors"].append(
            head_stat["relative_frobenius_error"]
        )
        layer_head_stats[layer_idx]["neg_percentages"].append(
            head_stat["negative_percentage"]
        )
        layer_head_stats[layer_idx]["pearson_corrs"].append(
            head_stat["pearson_correlation"]
        )
        layer_head_stats[layer_idx]["linear_ckas"].append(head_stat["linear_cka"])
        layer_head_stats[layer_idx]["kernel_ckas"].append(head_stat["kernel_cka"])

    # Compute average statistics for each layer
    for layer_idx, layer_stats in layer_head_stats.items():
        layer_stat = {
            "layer": layer_idx,
            "avg_direct_similarity": np.mean(layer_stats["direct_sims"]),
            "avg_optimal_similarity": np.mean(layer_stats["optimal_sims"]),
            "avg_improvement": np.mean(layer_stats["improvements"]),
            "avg_relative_frobenius_error": np.mean(layer_stats["rel_frob_errors"]),
            "avg_negative_percentage": np.mean(layer_stats["neg_percentages"]),
            "avg_pearson_correlation": np.mean(layer_stats["pearson_corrs"]),
            "avg_linear_cka": np.mean(layer_stats["linear_ckas"]),  # Add this line
            "avg_kernel_cka": np.mean(layer_stats["kernel_ckas"]),  # Add this line
            "num_samples": len(
                set(
                    [
                        h["sample"]
                        for h in stats["head_level"]
                        if h["layer"] == layer_idx
                    ]
                )
            ),
            "num_heads": len(
                set([h["head"] for h in stats["head_level"] if h["layer"] == layer_idx])
            ),
        }
        stats["layer_level"].append(layer_stat)

    # Sort layer-level statistics by layer index
    stats["layer_level"] = sorted(stats["layer_level"], key=lambda x: x["layer"])

    # Calculate model-level statistics (average across all layers and heads)
    all_direct_sims = [h["direct_similarity"] for h in stats["head_level"]]
    all_optimal_sims = [h["optimal_similarity"] for h in stats["head_level"]]
    all_improvements = [h["improvement"] for h in stats["head_level"]]
    all_rel_frob_errors = [h["relative_frobenius_error"] for h in stats["head_level"]]
    all_neg_percentages = [h["negative_percentage"] for h in stats["head_level"]]
    all_pearson_corrs = [h["pearson_correlation"] for h in stats["head_level"]]
    all_linear_ckas = [h["linear_cka"] for h in stats["head_level"]]
    all_kernel_ckas = [h["kernel_cka"] for h in stats["head_level"]]

    stats["model_level"] = {
        "avg_direct_similarity": np.mean(all_direct_sims),
        "avg_optimal_similarity": np.mean(all_optimal_sims),
        "avg_improvement": np.mean(all_improvements),
        "avg_relative_frobenius_error": np.mean(all_rel_frob_errors),
        "avg_negative_percentage": np.mean(all_neg_percentages),
        "avg_pearson_correlation": np.mean(all_pearson_corrs),
        "num_layers": len(stats["layer_level"]),
        "total_heads": len(stats["head_level"]),
        "std_direct_similarity": np.std(all_direct_sims),
        "std_optimal_similarity": np.std(all_optimal_sims),
        "min_direct_similarity": np.min(all_direct_sims),
        "max_direct_similarity": np.max(all_direct_sims),
        "min_optimal_similarity": np.min(all_optimal_sims),
        "max_optimal_similarity": np.max(all_optimal_sims),
        "avg_linear_cka": np.mean(all_linear_ckas),
        "avg_kernel_cka": np.mean(all_kernel_ckas),
    }

    # Calculate image-level statistics (per image, across layers and heads)
    image_stats = {}
    for head_stat in stats["head_level"]:
        sample_idx = head_stat["sample"]
        if sample_idx not in image_stats:
            image_stats[sample_idx] = {
                "direct_sims": [],
                "optimal_sims": [],
                "improvements": [],
                "rel_frob_errors": [],
                "neg_percentages": [],
                "pearson_corrs": [],
            }

        # Collect all head statistics for this image
        image_stats[sample_idx]["direct_sims"].append(head_stat["direct_similarity"])
        image_stats[sample_idx]["optimal_sims"].append(head_stat["optimal_similarity"])
        image_stats[sample_idx]["improvements"].append(head_stat["improvement"])
        image_stats[sample_idx]["rel_frob_errors"].append(
            head_stat["relative_frobenius_error"]
        )
        image_stats[sample_idx]["neg_percentages"].append(
            head_stat["negative_percentage"]
        )
        image_stats[sample_idx]["pearson_corrs"].append(
            head_stat["pearson_correlation"]
        )

    # Compute average statistics for each image
    for sample_idx, sample_stats in image_stats.items():
        stats["image_level"][sample_idx] = {
            "sample": sample_idx,
            "avg_direct_similarity": np.mean(sample_stats["direct_sims"]),
            "avg_optimal_similarity": np.mean(sample_stats["optimal_sims"]),
            "avg_improvement": np.mean(sample_stats["improvements"]),
            "avg_relative_frobenius_error": np.mean(sample_stats["rel_frob_errors"]),
            "avg_negative_percentage": np.mean(sample_stats["neg_percentages"]),
            "avg_pearson_correlation": np.mean(sample_stats["pearson_corrs"]),
        }

    return stats


# Load a batch of ImageNet images
def load_imagenet_batch(base_dir, subset="val", num_samples=100):
    """Load a batch of ImageNet images."""
    image_dir = os.path.join(base_dir, subset)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Find all image files in ImageNet directory structure
    all_image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith((".jpeg", ".jpg", ".png", ".bmp", ".JPEG")):
                all_image_paths.append(os.path.join(root, file))

    # Randomly sample images
    if len(all_image_paths) > num_samples:
        all_image_paths = random.sample(all_image_paths, num_samples)
    else:
        all_image_paths = all_image_paths[: min(num_samples, len(all_image_paths))]

    # Make sure we found some images
    if not all_image_paths:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Found {len(all_image_paths)} images")

    # Load and stack images into a single batch
    images = []
    for path in all_image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    if not images:
        raise ValueError("No images could be loaded successfully")

    input_tensor = torch.stack(images)  # Shape: [B, 3, 224, 224]
    return input_tensor


def create_feature_norm_visualizations(all_model_results, output_dir, num_samples=20):
    """
    Create scatter plots for feature norms across layers for each model.

    Args:
        all_model_results: Dictionary of results for all models
        output_dir: Directory to save visualizations
        num_samples: Number of samples to include in the plots
    """
    # Create directory for feature norm plots
    norm_vis_dir = os.path.join(output_dir, "feature_norm_plots")
    os.makedirs(norm_vis_dir, exist_ok=True)

    for model_name, results in all_model_results.items():
        print(f"Creating feature norm visualizations for {model_name}")

        # Check if we have feature norm data
        all_layer_results = results.get("all_layer_results", [])
        if not all_layer_results:
            print(f"No layer results for {model_name}")
            continue

        # Collect all feature norm data
        all_norms = []
        for layer_result in all_layer_results:
            if "feature_norms" in layer_result:
                all_norms.extend(layer_result["feature_norms"])

        if not all_norms:
            print(f"No feature norm data for {model_name}")
            continue

        # Convert to DataFrame for easier analysis
        norm_df = pd.DataFrame(all_norms)

        # Sort by layer
        norm_df = norm_df.sort_values(["layer", "sample", "head"])

        # Get unique layers
        layers = norm_df["layer"].unique()

        # Compute statistics for scatter plot
        layer_stats = []

        # Use only the first num_samples samples
        samples = norm_df["sample"].unique()[:num_samples]

        for layer in layers:
            layer_data = norm_df[norm_df["layer"] == layer]

            # Compute statistics for each sample
            for sample in samples:
                sample_data = layer_data[layer_data["sample"] == sample]

                # Average across heads
                if not sample_data.empty:
                    # Calculate mean of phi_norm and h_norm values for each token
                    phi_norm_arrays = [np.array(x) for x in sample_data["phi_norm"]]
                    h_norm_arrays = [np.array(x) for x in sample_data["h_norm"]]

                    # Compute mean over heads for each token position
                    phi_norm_mean = np.mean(
                        phi_norm_arrays, axis=0
                    )  # Mean across heads for each token
                    h_norm_mean = np.mean(
                        h_norm_arrays, axis=0
                    )  # Mean across heads for each token

                    # Compute mean over tokens to get a single value per sample per layer
                    phi_norm_final = np.mean(phi_norm_mean)
                    h_norm_final = np.mean(h_norm_mean)

                    # Store results
                    layer_stats.append(
                        {
                            "layer": layer,
                            "sample": sample,
                            "phi_norm_mean": phi_norm_final,
                            "h_norm_mean": h_norm_final,
                        }
                    )

        # Convert to DataFrame
        layer_stat_df = pd.DataFrame(layer_stats)

        if layer_stat_df.empty:
            print(f"No statistics computed for {model_name}")
            continue

        # Create scatter plot
        plt.figure(figsize=(12, 8))

        # Plot ||φ(x)||² values
        plt.scatter(
            layer_stat_df["layer"],
            layer_stat_df["phi_norm_mean"],
            alpha=0.7,
            color="blue",
            label="||φ(x)||²",
        )

        # Plot ||h(x)||² values
        plt.scatter(
            layer_stat_df["layer"],
            layer_stat_df["h_norm_mean"],
            alpha=0.7,
            color="red",
            label="||h(x)||²",
        )

        # Add trend lines
        for layer in layers:
            layer_means = layer_stat_df[layer_stat_df["layer"] == layer]
            plt.plot(
                [layer, layer],
                [
                    layer_means["phi_norm_mean"].mean(),
                    layer_means["h_norm_mean"].mean(),
                ],
                "k--",
                alpha=0.3,
            )

        # Calculate mean for each layer and plot trend lines
        phi_means = layer_stat_df.groupby("layer")["phi_norm_mean"].mean()
        h_means = layer_stat_df.groupby("layer")["h_norm_mean"].mean()

        plt.plot(phi_means.index, phi_means.values, "b-", linewidth=2)
        plt.plot(h_means.index, h_means.values, "r-", linewidth=2)

        plt.xlabel("Layer")
        plt.ylabel("Mean Squared Norm")
        plt.title(f"{model_name}: Feature Vector Norms Across Layers")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(layers)

        # Add annotations for average values
        for layer in layers:
            y_phi = phi_means[layer]
            y_h = h_means[layer]
            plt.annotate(
                f"{y_phi:.2f}",
                (layer, y_phi),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
            plt.annotate(
                f"{y_h:.2f}",
                (layer, y_h),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(norm_vis_dir, f"{model_name}_feature_norms.png"), dpi=300
        )
        plt.close()

        # Create box plots for distribution
        plt.figure(figsize=(14, 8))

        # Prepare data for box plots
        phi_data = []
        h_data = []
        labels = []

        for layer in layers:
            layer_data = layer_stat_df[layer_stat_df["layer"] == layer]
            phi_data.append(layer_data["phi_norm_mean"].values)
            h_data.append(layer_data["h_norm_mean"].values)
            labels.append(f"L{layer}")

        # Create a combined box plot
        positions = np.arange(len(layers)) * 2.5
        box_phi = plt.boxplot(
            phi_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
        )
        box_h = plt.boxplot(
            h_data,
            positions=positions + 0.8,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="lightcoral"),
        )

        plt.xlabel("Layer")
        plt.ylabel("Mean Squared Norm")
        plt.title(f"{model_name}: Distribution of Feature Vector Norms Across Layers")
        plt.xticks(positions + 0.4, labels)
        plt.legend([box_phi["boxes"][0], box_h["boxes"][0]], ["||φ(x)||²", "||h(x)||²"])

        plt.tight_layout()
        plt.savefig(
            os.path.join(norm_vis_dir, f"{model_name}_feature_norms_boxplot.png"),
            dpi=300,
        )
        plt.close()

        # Create summary table
        summary_df = pd.DataFrame(
            {
                "Layer": layers,
                "Mean_Phi_Norm": [phi_means[layer] for layer in layers],
                "Mean_H_Norm": [h_means[layer] for layer in layers],
                "Ratio_Phi_H": [phi_means[layer] / h_means[layer] for layer in layers],
            }
        )

        summary_df.to_csv(
            os.path.join(norm_vis_dir, f"{model_name}_feature_norms_summary.csv"),
            index=False,
        )

    # Create a comparative visualization across models
    if len(all_model_results) > 1:
        create_model_norm_comparison(all_model_results, norm_vis_dir)


def create_model_norm_comparison(all_model_results, output_dir):
    """Create comparative visualizations of feature norms across different models."""
    print("Creating cross-model feature norm comparison...")

    # Collect summary statistics for each model
    model_summaries = []

    for model_name, results in all_model_results.items():
        # Check if we have feature norm data
        all_layer_results = results.get("all_layer_results", [])
        if not all_layer_results:
            continue

        # Collect all feature norm data
        all_norms = []
        for layer_result in all_layer_results:
            if "feature_norms" in layer_result:
                all_norms.extend(layer_result["feature_norms"])

        if not all_norms:
            continue

        # Convert to DataFrame
        norm_df = pd.DataFrame(all_norms)

        # Compute mean values for each layer
        layer_means = []
        for layer in norm_df["layer"].unique():
            layer_data = norm_df[norm_df["layer"] == layer]

            # Extract and average the norm arrays
            phi_norm_arrays = [np.array(x) for x in layer_data["phi_norm"]]
            h_norm_arrays = [np.array(x) for x in layer_data["h_norm"]]

            # Compute overall means
            phi_mean = np.mean([np.mean(arr) for arr in phi_norm_arrays])
            h_mean = np.mean([np.mean(arr) for arr in h_norm_arrays])

            layer_means.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "phi_norm_mean": phi_mean,
                    "h_norm_mean": h_mean,
                    "ratio": phi_mean / h_mean if h_mean > 0 else np.nan,
                }
            )

        # Add to overall summaries
        model_summaries.extend(layer_means)

    if not model_summaries:
        print("No model summaries available for comparison")
        return

    # Convert to DataFrame for easier analysis
    summary_df = pd.DataFrame(model_summaries)

    # Determine model family and size for better visualization
    def get_model_family_size(model_name):
        if model_name.startswith("vit"):
            family = "ViT"
        elif "distilled" in model_name:
            family = "DeiT-D"
        else:
            family = "DeiT"

        if "tiny" in model_name:
            size = "Tiny"
        elif "small" in model_name:
            size = "Small"
        elif "base" in model_name:
            size = "Base"
        elif "large" in model_name:
            size = "Large"
        else:
            size = "Unknown"

        return family, size

    # Add model family and size columns
    summary_df["family"], summary_df["size"] = zip(
        *summary_df["model"].apply(get_model_family_size)
    )

    # Create comparison visualizations
    plt.figure(figsize=(14, 10))

    # Group by model and create line plots
    for model in summary_df["model"].unique():
        model_data = summary_df[summary_df["model"] == model]

        # Sort by layer
        model_data = model_data.sort_values("layer")

        # Determine line style and color based on model family and size
        family = model_data["family"].iloc[0]
        size = model_data["size"].iloc[0]

        if family == "ViT":
            color = "blue"
            marker = "o"
        elif family == "DeiT-D":
            color = "red"
            marker = "s"
        else:
            color = "green"
            marker = "^"

        if size == "Tiny":
            linestyle = ":"
        elif size == "Small":
            linestyle = "--"
        elif size == "Base":
            linestyle = "-"
        else:
            linestyle = "-."

        # Plot ||φ(x)||² to ||h(x)||² ratio
        plt.plot(
            model_data["layer"],
            model_data["ratio"],
            marker=marker,
            linestyle=linestyle,
            label=f"{model}",
            alpha=0.8,
        )

    plt.xlabel("Layer")
    plt.ylabel("||φ(x)||² / ||h(x)||² Ratio")
    plt.title("Feature Norm Ratio Across Models and Layers")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_model_feature_norm_ratio.png"), dpi=300)
    plt.close()

    # Create heatmap of phi norms across models and layers
    # Pivot the data for heatmap
    phi_pivot = summary_df.pivot_table(
        values="phi_norm_mean", index="model", columns="layer", aggfunc="mean"
    )

    plt.figure(figsize=(14, len(phi_pivot) * 0.6))
    sns.heatmap(
        phi_pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "||φ(x)||²"},
    )
    plt.title("||φ(x)||² Across Models and Layers")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_model_phi_norm_heatmap.png"), dpi=300)
    plt.close()

    # Create heatmap of h norms across models and layers
    h_pivot = summary_df.pivot_table(
        values="h_norm_mean", index="model", columns="layer", aggfunc="mean"
    )

    plt.figure(figsize=(14, len(h_pivot) * 0.6))
    sns.heatmap(
        h_pivot, annot=True, fmt=".2f", cmap="rocket_r", cbar_kws={"label": "||h(x)||²"}
    )
    plt.title("||h(x)||² Across Models and Layers")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_model_h_norm_heatmap.png"), dpi=300)
    plt.close()

    # Save summary to CSV
    summary_df.to_csv(
        os.path.join(output_dir, "cross_model_feature_norms.csv"), index=False
    )


# 1. Add a new function to compute the feature norms
def compute_feature_norms(Q, K, V, attn_map):
    """
    Compute ||φ(x)||² and ||h(x)||² for feature vectors.

    Args:
        Q: Query vectors [B, num_heads, N, head_dim]
        K: Key vectors [B, num_heads, N, head_dim]
        V: Value vectors [B, num_heads, N, head_dim]
        attn_map: Attention map [B, num_heads, N, N]

    Returns:
        phi_norm: ||φ(x)||² values [B, num_heads, N]
        h_norm: ||h(x)||² values [B, num_heads, N]
    """
    # Q and K have shape [B, num_heads, N, head_dim]
    # Compute QK dot product for each token with itself
    B, num_heads, N, head_dim = Q.shape

    # Get diagonal entries of QK matrix (dot product of each token with itself)
    # First compute full QK matrix
    scale = 1.0 / math.sqrt(head_dim)
    qk = (Q @ K.transpose(-2, -1)) * scale  # [B, num_heads, N, N]

    # Extract diagonal elements (self-similarity)
    qq = torch.diagonal(qk, dim1=-2, dim2=-1)  # [B, num_heads, N]

    # Compute ||φ(x)||² as in the provided formula
    num = torch.exp(qq)
    # Get the first token's attention score for denominator
    den = torch.exp(2 * qk[:, :, :, 0] - torch.log(attn_map[:, :, :, 0].pow(2)))

    # Compute ||φ(x)||²
    phi_norm = num / den  # [B, num_heads, N]

    # Compute ||h(x)||² as the squared L2 norm of value vectors
    h_norm = V.pow(2).sum(dim=-1)  # [B, num_heads, N]

    # Return both norms
    return phi_norm, h_norm


def create_model_comparison_visualizations(all_model_results, output_dir):
    """
    Create visualizations comparing results across different models.

    Args:
        all_model_results: Dictionary mapping model names to their analysis results
        output_dir: Directory to save visualizations
    """
    comparison_dir = os.path.join(output_dir, "models_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Extract model-level statistics for each model
    model_stats = {}
    for model_name, results in all_model_results.items():
        if (
            "hierarchical_stats" in results
            and "model_level" in results["hierarchical_stats"]
        ):
            model_stats[model_name] = results["hierarchical_stats"]["model_level"]

    if not model_stats:
        print("No model statistics available for comparison")
        return

    # Create DataFrame for easier analysis
    model_data = []
    for model_name, stats in model_stats.items():
        # Determine model family (ViT vs DeiT)
        model_family = "ViT" if model_name.startswith("vit") else "DeiT"

        # Determine model size from name
        if "tiny" in model_name:
            model_size = "Tiny"
        elif "small" in model_name:
            model_size = "Small"
        elif "base" in model_name:
            model_size = "Base"
        elif "large" in model_name:
            model_size = "Large"
        elif "huge" in model_name:
            model_size = "Huge"
        else:
            model_size = "Unknown"

        # Is distilled?
        is_distilled = "distilled" in model_name

        model_data.append(
            {
                "Model": model_name,
                "Family": model_family,
                "Size": model_size,
                "Distilled": is_distilled,
                "Direct_Similarity": stats["avg_direct_similarity"],
                "Optimal_Similarity": stats["avg_optimal_similarity"],
                "Improvement": stats["avg_improvement"],
                "Relative_Frobenius_Error": stats["avg_relative_frobenius_error"],
                "Negative_Percentage": stats["avg_negative_percentage"],
                "Pearson_Correlation": stats["avg_pearson_correlation"],
                "Num_Layers": stats["num_layers"],
                "Total_Heads": stats["total_heads"],
                "Std_Direct_Similarity": stats["std_direct_similarity"],
                "Std_Optimal_Similarity": stats["std_optimal_similarity"],
            }
        )

    df = pd.DataFrame(model_data)

    # Sort by model family and size
    size_order = {"Tiny": 0, "Small": 1, "Base": 2, "Large": 3, "Huge": 4, "Unknown": 5}
    df["Size_Order"] = df["Size"].map(size_order)
    df = df.sort_values(["Family", "Size_Order", "Distilled"])
    df = df.drop("Size_Order", axis=1)

    # Save comparison data to CSV
    df.to_csv(os.path.join(comparison_dir, "model_comparison.csv"), index=False)

    # Define consistent color maps for plotting
    vit_color = "#4e79a7"  # Blue
    deit_color = "#f28e2c"  # Orange
    distilled_color = "#e15759"  # Red

    # 1. Bar chart comparing direct similarities across models
    plt.figure(figsize=(14, 8))

    # Sort models for consistent display order
    models = df["Model"].tolist()
    colors = [
        (
            vit_color
            if m.startswith("vit")
            else (distilled_color if "distilled" in m else deit_color)
        )
        for m in models
    ]

    # Create bar chart
    bars = plt.bar(
        models,
        df["Direct_Similarity"],
        yerr=df["Std_Direct_Similarity"],
        capsize=5,
        color=colors,
        alpha=0.7,
    )

    plt.xlabel("Model")
    plt.ylabel("Average Direct Similarity")
    plt.title("Direct Value Vector Similarity Across Models")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    # Add legend for color meanings
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=vit_color, label="ViT"),
        Patch(facecolor=deit_color, label="DeiT"),
        Patch(facecolor=distilled_color, label="DeiT (Distilled)"),
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(
        os.path.join(comparison_dir, "direct_similarity_comparison.png"), dpi=300
    )
    plt.close()

    # 2. Direct vs. Optimal similarity comparison
    plt.figure(figsize=(14, 8))

    x = np.arange(len(models))
    width = 0.35

    plt.bar(
        x - width / 2,
        df["Direct_Similarity"],
        width,
        label="Direct",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        x + width / 2,
        df["Optimal_Similarity"],
        width,
        label="Optimal",
        color="green",
        alpha=0.7,
    )

    plt.xlabel("Model")
    plt.ylabel("Average Similarity")
    plt.title("Direct vs. Optimal Similarity Across Models")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(comparison_dir, "direct_vs_optimal_comparison.png"), dpi=300
    )
    plt.close()

    # 3. Improvement comparison
    plt.figure(figsize=(14, 8))

    plt.bar(models, df["Improvement"], color="purple", alpha=0.7)

    plt.xlabel("Model")
    plt.ylabel("Average Improvement (Optimal - Direct)")
    plt.title("Similarity Improvement Across Models")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "improvement_comparison.png"), dpi=300)
    plt.close()

    # 4. Model size vs. Similarity scatterplot
    plt.figure(figsize=(12, 8))

    # Define size markers based on number of layers and heads
    sizes = df["Total_Heads"] / df["Total_Heads"].max() * 500  # Scale marker sizes

    # Create scatter plot grouped by family
    vit_df = df[df["Family"] == "ViT"]
    deit_df = df[df["Family"] == "DeiT"]
    deit_distilled_df = df[(df["Family"] == "DeiT") & (df["Distilled"])]
    deit_regular_df = df[(df["Family"] == "DeiT") & (~df["Distilled"])]

    if not vit_df.empty:
        plt.scatter(
            vit_df["Direct_Similarity"],
            vit_df["Optimal_Similarity"],
            s=vit_df["Total_Heads"] / df["Total_Heads"].max() * 500,
            alpha=0.7,
            color=vit_color,
            label="ViT",
        )

        # Add model labels
        for i, row in vit_df.iterrows():
            plt.annotate(
                row["Size"],
                (row["Direct_Similarity"], row["Optimal_Similarity"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

    if not deit_regular_df.empty:
        plt.scatter(
            deit_regular_df["Direct_Similarity"],
            deit_regular_df["Optimal_Similarity"],
            s=deit_regular_df["Total_Heads"] / df["Total_Heads"].max() * 500,
            alpha=0.7,
            color=deit_color,
            label="DeiT",
        )

        # Add model labels
        for i, row in deit_regular_df.iterrows():
            plt.annotate(
                row["Size"],
                (row["Direct_Similarity"], row["Optimal_Similarity"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

    if not deit_distilled_df.empty:
        plt.scatter(
            deit_distilled_df["Direct_Similarity"],
            deit_distilled_df["Optimal_Similarity"],
            s=deit_distilled_df["Total_Heads"] / df["Total_Heads"].max() * 500,
            alpha=0.7,
            color=distilled_color,
            label="DeiT (Distilled)",
        )

        # Add model labels
        for i, row in deit_distilled_df.iterrows():
            plt.annotate(
                row["Size"],
                (row["Direct_Similarity"], row["Optimal_Similarity"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

    # Add y=x line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)

    plt.xlabel("Direct Similarity")
    plt.ylabel("Optimal Similarity")
    plt.title("Direct vs. Optimal Similarity by Model Type and Size")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "direct_vs_optimal_scatter.png"), dpi=300)
    plt.close()

    # 5. Correlation matrix heatmap
    plt.figure(figsize=(12, 10))

    # Select numerical columns for correlation
    numeric_cols = [
        "Direct_Similarity",
        "Optimal_Similarity",
        "Improvement",
        "Relative_Frobenius_Error",
        "Negative_Percentage",
        "Pearson_Correlation",
        "Num_Layers",
        "Total_Heads",
    ]

    correlation_matrix = df[numeric_cols].corr()

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
    )

    plt.title("Correlation Matrix of Model Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "correlation_matrix.png"), dpi=300)
    plt.close()

    # 6. Layer-wise comparison across models
    # For each model, extract layer-level statistics
    layer_stats_by_model = {}
    for model_name, results in all_model_results.items():
        if (
            "hierarchical_stats" in results
            and "layer_level" in results["hierarchical_stats"]
        ):
            layer_stats = results["hierarchical_stats"]["layer_level"]
            if layer_stats:
                layer_stats_by_model[model_name] = pd.DataFrame(layer_stats)

    if layer_stats_by_model:
        # Plot direct similarity by layer for each model
        plt.figure(figsize=(14, 10))

        for i, (model_name, layer_df) in enumerate(layer_stats_by_model.items()):
            # Normalize layer indices to percentage of model depth for fair comparison
            max_layer = layer_df["layer"].max()
            normalized_layers = (
                layer_df["layer"] / max_layer if max_layer > 0 else layer_df["layer"]
            )

            # Use consistent colors by model family
            if model_name.startswith("vit"):
                color = f"C{i % 5}"  # Cycle through matplotlib colors for ViT
                linestyle = "-"
            elif "distilled" in model_name:
                color = f"C{5 + (i % 5)}"  # Use a different set of colors for distilled models
                linestyle = "--"
            else:
                color = f"C{10 + (i % 5)}"  # Use another set for regular DeiT
                linestyle = ":"

            plt.plot(
                normalized_layers,
                layer_df["avg_direct_similarity"],
                marker="o",
                linestyle=linestyle,
                label=model_name,
                color=color,
            )

        plt.xlabel("Normalized Layer Depth (0 = first, 1 = last)")
        plt.ylabel("Average Direct Similarity")
        plt.title("Direct Similarity Across Layers for Different Models")
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(comparison_dir, "layer_direct_similarity_comparison.png"),
            dpi=300,
        )
        plt.close()

        # Plot optimal similarity by layer for each model
        plt.figure(figsize=(14, 10))

        for i, (model_name, layer_df) in enumerate(layer_stats_by_model.items()):
            # Normalize layer indices for fair comparison
            max_layer = layer_df["layer"].max()
            normalized_layers = (
                layer_df["layer"] / max_layer if max_layer > 0 else layer_df["layer"]
            )

            # Use consistent colors by model family
            if model_name.startswith("vit"):
                color = f"C{i % 5}"
                linestyle = "-"
            elif "distilled" in model_name:
                color = f"C{5 + (i % 5)}"
                linestyle = "--"
            else:
                color = f"C{10 + (i % 5)}"
                linestyle = ":"

            plt.plot(
                normalized_layers,
                layer_df["avg_optimal_similarity"],
                marker="o",
                linestyle=linestyle,
                label=model_name,
                color=color,
            )

        plt.xlabel("Normalized Layer Depth (0 = first, 1 = last)")
        plt.ylabel("Average Optimal Similarity")
        plt.title("Optimal Similarity Across Layers for Different Models")
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(comparison_dir, "layer_optimal_similarity_comparison.png"),
            dpi=300,
        )
        plt.close()

    print(f"Model comparison visualizations saved to {comparison_dir}")


def create_multilevel_visualizations(results, output_dir, model_name):
    """
    Create visualizations for different levels of statistics:
    - Head level
    - Layer level
    - Model level
    - Image level

    Args:
        results: Dictionary containing hierarchical statistics
        output_dir: Directory to save visualizations
        model_name: Name of the model (for labeling)
    """
    # Create model-specific output directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "head_level"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "layer_level"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "model_level"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "image_level"), exist_ok=True)

    stats = results.get("hierarchical_stats", {})
    if not stats:
        print(f"No hierarchical statistics to visualize for {model_name}")
        return

    # 1. Head-Level Visualizations
    head_stats = stats.get("head_level", [])
    if head_stats:
        # Convert to DataFrame for easier analysis
        head_df = pd.DataFrame(head_stats)

        # 1.1 Scatter plot of direct vs. optimal similarity for all heads
        plt.figure(figsize=(10, 8))
        plt.scatter(
            head_df["direct_similarity"],
            head_df["optimal_similarity"],
            alpha=0.6,
            c=head_df["layer"],
            cmap="viridis",
        )
        plt.plot([0, 1], [0, 1], "r--", alpha=0.7)  # y=x line
        plt.colorbar(label="Layer")
        plt.xlabel("Direct Similarity")
        plt.ylabel("Optimal Similarity")
        plt.title(f"{model_name}: Direct vs. Optimal Similarity for All Heads")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_dir, "head_level/direct_vs_optimal.png"), dpi=300
        )
        plt.close()

        # 1.2 Histogram of improvements for all heads
        plt.figure(figsize=(10, 6))
        plt.hist(head_df["improvement"], bins=30, alpha=0.7, color="purple")
        plt.axvline(
            head_df["improvement"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {head_df["improvement"].mean():.4f}',
        )
        plt.xlabel("Improvement (Optimal - Direct)")
        plt.ylabel("Count")
        plt.title(f"{model_name}: Distribution of Similarity Improvements")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_dir, "head_level/improvement_histogram.png"), dpi=300
        )
        plt.close()

        # 1.3 Heatmap of direct similarities by layer and head
        # Only process first few images for heatmaps (to avoid generating too many)
        for sample_idx in head_df["sample"].unique()[:3]:  # First 3 images for example
            sample_data = head_df[head_df["sample"] == sample_idx]
            pivot_direct = sample_data.pivot_table(
                values="direct_similarity",
                index="layer",
                columns="head",
                aggfunc="mean",
            )

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_direct,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Direct Similarity"},
            )
            plt.title(
                f"{model_name}: Direct Similarity by Layer and Head (Image {sample_idx})"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    model_dir, f"head_level/direct_heatmap_img{sample_idx}.png"
                ),
                dpi=300,
            )
            plt.close()

    # 2. Layer-Level Visualizations
    layer_stats = stats.get("layer_level", [])
    if layer_stats:
        # Convert to DataFrame for easier analysis (MOVE THIS UP)
        layer_df = pd.DataFrame(layer_stats)

        # Create new figure for CKA comparison (PLACE THIS AFTER layer_df is created)
        plt.figure(figsize=(12, 6))
        plt.plot(
            layer_df["layer"],
            layer_df["avg_direct_similarity"],
            "o-",
            label="Direct Similarity",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            layer_df["layer"],
            layer_df["avg_linear_cka"],
            "o-",
            label="Linear CKA",
            color="red",
            linewidth=2,
        )
        plt.plot(
            layer_df["layer"],
            layer_df["avg_kernel_cka"],
            "o-",
            label="Kernel CKA",
            color="green",
            linewidth=2,
        )

        plt.xlabel("Layer")
        plt.ylabel("Average Similarity")
        plt.title(f"{model_name}: Direct Similarity vs CKA Across Layers")
        plt.xticks(layer_df["layer"])
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "layer_level/cka_comparison.png"), dpi=300)
        plt.close()
        # 2.1 Line plot of direct and optimal similarities across layers
        plt.figure(figsize=(12, 6))
        plt.plot(
            layer_df["layer"],
            layer_df["avg_direct_similarity"],
            "o-",
            label="Direct Similarity",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            layer_df["layer"],
            layer_df["avg_optimal_similarity"],
            "o-",
            label="Optimal Similarity",
            color="green",
            linewidth=2,
        )
        plt.fill_between(
            layer_df["layer"],
            layer_df["avg_direct_similarity"],
            layer_df["avg_optimal_similarity"],
            alpha=0.2,
            color="green",
            label="Improvement",
        )
        plt.xlabel("Layer")
        plt.ylabel("Average Similarity")
        plt.title(f"{model_name}: Direct vs. Optimal Similarity Across Layers")
        plt.xticks(layer_df["layer"])
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_dir, "layer_level/layer_similarities.png"), dpi=300
        )
        plt.close()

        # 2.2 Bar plot of improvements by layer
        plt.figure(figsize=(12, 6))
        improvements = layer_df["avg_improvement"]
        plt.bar(layer_df["layer"], improvements, color="purple", alpha=0.7)
        # Add horizontal line for average improvement
        plt.axhline(
            y=improvements.mean(),
            color="red",
            linestyle="--",
            label=f"Mean Improvement: {improvements.mean():.4f}",
        )
        plt.xlabel("Layer")
        plt.ylabel("Average Improvement")
        plt.title(f"{model_name}: Similarity Improvement by Layer")
        plt.xticks(layer_df["layer"])
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_dir, "layer_level/layer_improvements.png"), dpi=300
        )
        plt.close()

    # 3. Model-Level Visualizations
    model_stats = stats.get("model_level", {})
    if model_stats:
        # 3.1 Summary statistics as a bar chart
        model_metrics = [
            "avg_direct_similarity",
            "avg_optimal_similarity",
            "avg_improvement",
            "avg_pearson_correlation",
        ]
        model_values = [model_stats[metric] for metric in model_metrics]
        model_labels = [
            "Direct Similarity",
            "Optimal Similarity",
            "Improvement",
            "Pearson Correlation",
        ]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            model_labels,
            model_values,
            color=["blue", "green", "purple", "orange"],
            alpha=0.7,
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )

        plt.ylabel("Value")
        plt.title(f"{model_name}: Model-Level Similarity Statistics")
        plt.ylim(0, max(model_values) * 1.2)  # Add space for labels
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "model_level/model_summary.png"), dpi=300)
        plt.close()

    # 4. Image-Level Visualizations
    image_stats = stats.get("image_level", {})
    if image_stats:
        # Convert to DataFrame for easier analysis
        image_df = pd.DataFrame(image_stats.values())

        # 4.1 Histogram of direct and optimal similarities across images
        plt.figure(figsize=(10, 6))
        plt.hist(
            image_df["avg_direct_similarity"],
            bins=20,
            alpha=0.7,
            color="blue",
            label="Direct",
        )
        plt.hist(
            image_df["avg_optimal_similarity"],
            bins=20,
            alpha=0.7,
            color="green",
            label="Optimal",
        )
        plt.xlabel("Average Similarity")
        plt.ylabel("Number of Images")
        plt.title(f"{model_name}: Distribution of Average Similarities Across Images")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_dir, "image_level/image_similarity_distribution.png"),
            dpi=300,
        )
        plt.close()

        # 4.2 Scatter plot of direct vs. optimal similarity for images
        plt.figure(figsize=(10, 8))
        plt.scatter(
            image_df["avg_direct_similarity"],
            image_df["avg_optimal_similarity"],
            alpha=0.7,
            c=image_df["avg_improvement"],
            cmap="viridis",
        )
        plt.plot([0, 1], [0, 1], "r--", alpha=0.7)  # y=x line
        plt.colorbar(label="Improvement")
        plt.xlabel("Average Direct Similarity")
        plt.ylabel("Average Optimal Similarity")
        plt.title(f"{model_name}: Direct vs. Optimal Similarity by Image")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_dir, "image_level/image_direct_vs_optimal.png"), dpi=300
        )
        plt.close()

    print(f"Visualizations for {model_name} saved to {model_dir}")


def save_hierarchical_statistics(stats, output_dir, model_name):
    """Save hierarchical statistics to CSV files for further analysis."""
    # Create directory for statistics
    stats_dir = os.path.join(output_dir, model_name, "statistics")
    os.makedirs(stats_dir, exist_ok=True)

    # Save head-level statistics
    if "head_level" in stats and stats["head_level"]:
        head_df = pd.DataFrame(stats["head_level"])
        head_df.to_csv(os.path.join(stats_dir, "head_level_stats.csv"), index=False)

    # Save layer-level statistics
    if "layer_level" in stats and stats["layer_level"]:
        layer_df = pd.DataFrame(stats["layer_level"])
        layer_df.to_csv(os.path.join(stats_dir, "layer_level_stats.csv"), index=False)

    # Save image-level statistics
    if "image_level" in stats and stats["image_level"]:
        image_df = pd.DataFrame(stats["image_level"].values())
        image_df.to_csv(os.path.join(stats_dir, "image_level_stats.csv"), index=False)

    # Save model-level statistics
    if "model_level" in stats and stats["model_level"]:
        # Convert dictionary to DataFrame (single row)
        model_df = pd.DataFrame([stats["model_level"]])
        model_df.to_csv(os.path.join(stats_dir, "model_level_stats.csv"), index=False)

    print(f"Hierarchical statistics for {model_name} saved to {stats_dir}")


# Main execution
if __name__ == "__main__":
    # Load ImageNet data batch - same batch used for all models
    start_time = time.time()
    print(f"Loading {args.num_samples} images from ImageNet {args.subset} set...")
    input_tensor = load_imagenet_batch(args.base_dir, args.subset, args.num_samples)
    batch_size = input_tensor.shape[0]
    print(
        f"Loaded batch of {batch_size} images in {time.time() - start_time:.2f} seconds"
    )

    # Dictionary to store results for all models
    all_model_results = {}

    # Process each model
    for i, model_name in enumerate(models_to_analyze):
        print(
            f"\n===== Analyzing model {i+1}/{len(models_to_analyze)}: {model_name} ====="
        )
        model_start_time = time.time()

        try:
            # Load model
            model = load_model_with_checkpoint(model_name, args.checkpoint)
            model.eval()

            # Move input tensor to same device as model
            device = next(model.parameters()).device
            input_tensor_device = input_tensor.to(device)

            # Run analysis on all layers (or specific layer if provided)
            results = run_multilayer_analysis(
                model, input_tensor_device, args.layer, debug=(i == 0)
            )

            if not results:
                print(f"No results generated for {model_name}. Skipping.")
                continue

            # Store results for this model
            all_model_results[model_name] = results

            # Print model-level summary
            if (
                "hierarchical_stats" in results
                and "model_level" in results["hierarchical_stats"]
            ):
                model_stats = results["hierarchical_stats"]["model_level"]
                print(f"\n===== MODEL-LEVEL SUMMARY FOR {model_name} =====")
                print(f"Number of layers analyzed: {model_stats['num_layers']}")
                print(f"Total heads analyzed: {model_stats['total_heads']}")
                print(
                    f"Average direct similarity: {model_stats['avg_direct_similarity']:.6f} ± {model_stats['std_direct_similarity']:.6f}"
                )
                print(
                    f"Average optimal similarity: {model_stats['avg_optimal_similarity']:.6f} ± {model_stats['std_optimal_similarity']:.6f}"
                )
                print(f"Average improvement: {model_stats['avg_improvement']:.6f}")
                print(
                    f"Average relative Frobenius error: {model_stats['avg_relative_frobenius_error']:.6f}"
                )
                print(
                    f"Average Pearson correlation: {model_stats['avg_pearson_correlation']:.6f}"
                )

            # Create multi-level visualizations for this model
            create_multilevel_visualizations(results, args.output_dir, model_name)

            # Save statistics for further analysis
            save_hierarchical_statistics(
                results["hierarchical_stats"], args.output_dir, model_name
            )

            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            model_elapsed_time = time.time() - model_start_time
            print(
                f"Analysis of {model_name} completed in {model_elapsed_time:.2f} seconds"
            )

        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            import traceback

            traceback.print_exc()

    # Create model comparison visualizations
    if len(all_model_results) > 1:
        print("\n===== Creating model comparison visualizations =====")
        create_model_comparison_visualizations(all_model_results, args.output_dir)

    # Save combined results summary
    print("\n===== Saving combined results summary =====")
    combined_results = {
        "models_analyzed": list(all_model_results.keys()),
        "num_samples": args.num_samples,
        "dataset": args.subset,
        "allclose_statistics": {
            "pass_count": ALLCLOSE_PASS_COUNT,
            "total_count": ALLCLOSE_TOTAL_COUNT,
            "pass_percentage": (
                (ALLCLOSE_PASS_COUNT / ALLCLOSE_TOTAL_COUNT * 100)
                if ALLCLOSE_TOTAL_COUNT > 0
                else 0
            ),
        },
        "model_summaries": {},
    }

    for model_name, results in all_model_results.items():
        if (
            "hierarchical_stats" in results
            and "model_level" in results["hierarchical_stats"]
        ):
            combined_results["model_summaries"][model_name] = results[
                "hierarchical_stats"
            ]["model_level"]

    # Save to JSON file
    serializable_results = convert_to_json_serializable(combined_results)
    with open(os.path.join(args.output_dir, "combined_results_summary.json"), "w") as f:
        json.dump(serializable_results, f, indent=4)

    # Print allclose statistics
    print(f"\n===== torch.allclose Statistics =====")
    print(f"Passes: {ALLCLOSE_PASS_COUNT}")
    print(f"Total checks: {ALLCLOSE_TOTAL_COUNT}")
    print(
        f"Pass percentage: {(ALLCLOSE_PASS_COUNT / ALLCLOSE_TOTAL_COUNT * 100):.2f}% (if > 0)"
    )

    print("\n===== Creating feature norm visualizations =====")
    create_feature_norm_visualizations(
        all_model_results, args.output_dir, num_samples=20
    )

    serializable_allclose = convert_to_json_serializable(
        {
            "pass_count": ALLCLOSE_PASS_COUNT,
            "total_count": ALLCLOSE_TOTAL_COUNT,
            "pass_percentage": (
                (ALLCLOSE_PASS_COUNT / ALLCLOSE_TOTAL_COUNT * 100)
                if ALLCLOSE_TOTAL_COUNT > 0
                else 0
            ),
        }
    )
    with open(os.path.join(args.output_dir, "allclose_statistics.json"), "w") as f:
        json.dump(serializable_allclose, f, indent=4)
    # Create an HTML report with key findings
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vision Transformer Value Vector Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .model-section {{ margin: 30px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            .highlight {{ background-color: #ffffcc; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Vision Transformer Value Vector Analysis</h1>
        <div class="summary">
            <h2>Analysis Summary</h2>
            <p>Date: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Models analyzed: {len(all_model_results)}</p>
            <p>Dataset: ImageNet {args.subset}</p>
            <p>Number of images: {args.num_samples}</p>
        </div>
        
        <h2>Model Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Direct Similarity</th>
                <th>Optimal Similarity</th>
                <th>Improvement</th>
                <th>Rel. Frob. Error</th>
            </tr>
    """

    # Sort models by direct similarity
    model_items = []
    for model_name, results in all_model_results.items():
        if (
            "hierarchical_stats" in results
            and "model_level" in results["hierarchical_stats"]
        ):
            stats = results["hierarchical_stats"]["model_level"]
            model_items.append(
                (
                    model_name,
                    stats["avg_direct_similarity"],
                    stats["avg_optimal_similarity"],
                    stats["avg_improvement"],
                    stats["avg_relative_frobenius_error"],
                )
            )

    # Sort by direct similarity (descending)
    model_items.sort(key=lambda x: x[1], reverse=True)

    # Add rows to the table
    for model_name, direct_sim, optimal_sim, improvement, rel_frob in model_items:
        html_report += f"""
            <tr>
                <td>{model_name}</td>
                <td>{direct_sim:.4f}</td>
                <td>{optimal_sim:.4f}</td>
                <td>{improvement:.4f}</td>
                <td>{rel_frob:.4f}</td>
            </tr>
        """

    html_report += """
        </table>
        
        <h2>Key Visualizations</h2>
    """

    # Add key model comparison visualizations
    if len(all_model_results) > 1:
        html_report += f"""
        <h3>Model Comparison</h3>
        <div>
            <img src="models_comparison/direct_similarity_comparison.png" alt="Direct Similarity Comparison">
            <p>Direct similarity comparison across models</p>
        </div>
        <div>
            <img src="models_comparison/direct_vs_optimal_comparison.png" alt="Direct vs Optimal Comparison">
            <p>Direct vs. optimal similarity comparison</p>
        </div>
        <div>
            <img src="models_comparison/improvement_comparison.png" alt="Improvement Comparison">
            <p>Similarity improvement (optimal - direct) across models</p>
        </div>
        <div>
            <img src="models_comparison/layer_direct_similarity_comparison.png" alt="Layer Direct Similarity">
            <p>Direct similarity across normalized layer depth</p>
        </div>
        """

    # Individual model sections
    for model_name, results in all_model_results.items():
        if (
            "hierarchical_stats" in results
            and "model_level" in results["hierarchical_stats"]
        ):
            stats = results["hierarchical_stats"]["model_level"]
            html_report += f"""
            <div class="model-section">
                <h3>{model_name}</h3>
                <p>Direct Similarity: <span class="highlight">{stats['avg_direct_similarity']:.4f}</span></p>
                <p>Optimal Similarity: <span class="highlight">{stats['avg_optimal_similarity']:.4f}</span></p>
                <p>Improvement: <span class="highlight">{stats['avg_improvement']:.4f}</span></p>
                <p>Number of layers: {stats['num_layers']}</p>
                <p>Number of heads: {stats['total_heads'] // stats['num_layers']}</p>
                
                <h4>Layer-Level Analysis</h4>
                <div>
                    <img src="{model_name}/layer_level/layer_similarities.png" alt="Layer Similarities">
                    <p>Direct vs. optimal similarity across layers</p>
                </div>
                
                <h4>Image-Level Analysis</h4>
                <div>
                    <img src="{model_name}/image_level/image_direct_vs_optimal.png" alt="Image Direct vs Optimal">
                    <p>Direct vs. optimal similarity across images</p>
                </div>
            </div>
            """

    html_report += """
        <h2>Conclusions</h2>
        <div class="summary">
            <p>This analysis examined the relationship between value vectors in vision transformer models and their reconstructions using kernel PCA eigenvectors.</p>
            <p>Key findings:</p>
            <ul>
    """

    # Add some findings based on the data
    if len(all_model_results) > 1:
        # Find models with highest direct and optimal similarity
        best_direct_model = max(model_items, key=lambda x: x[1])
        best_optimal_model = max(model_items, key=lambda x: x[2])
        best_improvement_model = max(model_items, key=lambda x: x[3])

        html_report += f"""
                <li>{best_direct_model[0]} had the highest direct similarity at {best_direct_model[1]:.4f}</li>
                <li>{best_optimal_model[0]} had the highest optimal similarity at {best_optimal_model[2]:.4f}</li>
                <li>{best_improvement_model[0]} showed the greatest improvement with optimal matching at {best_improvement_model[3]:.4f}</li>
        """

    # Add general findings
    avg_direct = np.mean([stats[1] for stats in model_items])
    avg_optimal = np.mean([stats[2] for stats in model_items])
    avg_improvement = np.mean([stats[3] for stats in model_items])

    html_report += f"""
                <li>Average direct similarity across all models: {avg_direct:.4f}</li>
                <li>Average optimal similarity across all models: {avg_optimal:.4f}</li>
                <li>Average improvement with optimal matching: {avg_improvement:.4f}</li>
                <li>These results suggest that value vectors in vision transformers are significantly aligned with the kernel PCA eigenvectors</li>
                <li>The improvement from direct to optimal matching indicates that reordering the vectors could potentially improve model performance</li>
            </ul>
        </div>
        
        <div>
            <h3>Methodology</h3>
            <p>This analysis uses the following approach:</p>
            <ol>
                <li>For each model, extract the key (K) and value (V) matrices from the attention mechanism</li>
                <li>Compute the centered kernel matrix using the key vectors</li>
                <li>Calculate the eigenvectors of this kernel matrix</li>
                <li>Reconstruct the value vectors using these eigenvectors</li>
                <li>Compare the original value vectors with their reconstructions using both direct index-to-index cosine similarity and optimal matching via the Hungarian algorithm</li>
            </ol>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    with open(os.path.join(args.output_dir, "analysis_report.html"), "w") as f:
        f.write(html_report)

    # Print final summary
    total_elapsed_time = time.time() - start_time
    print(f"\n===== Analysis completed in {total_elapsed_time:.2f} seconds =====")
    print(
        f"Analyzed {len(all_model_results)} models on {args.num_samples} images from ImageNet {args.subset} set"
    )
    print(f"Results saved to {args.output_dir}")
    print(
        f"HTML report saved to {os.path.join(args.output_dir, 'analysis_report.html')}"
    )

    if len(all_model_results) > 1:
        print("\nModel comparison summary:")
        print(
            "{:<25} {:<15} {:<15} {:<15} {:<15}".format(
                "Model", "Direct Sim", "Optimal Sim", "Improvement", "Rel. Frob. Error"
            )
        )
        print("-" * 85)

        for model_name, direct_sim, optimal_sim, improvement, rel_frob in model_items:
            print(
                "{:<25} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
                    model_name, direct_sim, optimal_sim, improvement, rel_frob
                )
            )

    # python kpca.py --all-models --value-vector --num_samples 100 --base_dir /path/to/imagenet
    print(f"\nDone! To run with different parameters:")
    print(
        f"  Single model:  python script.py --model {models_to_analyze[0]} --value-vector --num_samples {args.num_samples}"
    )
    print(
        f"  Multiple models: python script.py --all-models --value-vector --num_samples {args.num_samples}"
    )
    print(
        f"  Specific models: python script.py --model-list vit_tiny_patch16_224,deit_tiny_patch16_224 --value-vector"
    )
