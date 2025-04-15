import torch
import timm
import torch.nn.functional as F
from torchvision import datasets, transforms
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
import torch
import torch.backends.cudnn as cudnn
import yaml

# Initialize global counters for allclose statistics
ALLCLOSE_PASS_COUNT = 0
ALLCLOSE_TOTAL_COUNT = 0

# set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

# Function to load config from YAML file
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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

# source: https://github.com/jayroxis/CKA-similarity
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
                Q_bh = Q[b, h]  # [N, head_dim]

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
def load_imagenet_batch(base_dir, num_samples=100):
    """Load a batch of ImageNet images from validation set."""
    # Use validation folder for testing
    val_dir = os.path.join(base_dir, 'val')
    if not os.path.exists(val_dir):
        print(f"Warning: Validation directory {val_dir} not found. Using main directory.")
        val_dir = base_dir

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Use ImageFolder to load the data properly
    dataset = datasets.ImageFolder(val_dir, transform=transform)

    # Sample the dataset
    indices = list(range(len(dataset)))
    if len(indices) > num_samples:
        indices = random.sample(indices, num_samples)
    else:
        print(f"Warning: Requested {num_samples} samples but only found {len(indices)} images.")
        num_samples = len(indices)
    
    # Create dataloader with the sampled indices
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, sampler=sampler)

    # Get the batch of images
    for input_tensor, _ in loader:
        print(f"Loaded {input_tensor.shape[0]} images with shape: {input_tensor.shape}")
        return input_tensor  # Return the first batch
    
    # If no images could be loaded
    raise ValueError("No images could be loaded successfully")


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


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    image_dir = config.get('dataset_path')
    num_samples = 100
    model_name = config.get('model', 'vit_tiny_patch16_224')
    block_idx = None # all layers
    seed = config.get('seed', 0)
    output_dir = "./value_vectors_outputs"
    
    # Set random seed
    set_random_seed(seed)
    
    print(f"Configuration:")
    print(f"  Dataset path: {image_dir}")
    print(f"  Number of samples: {num_samples}")
    print(f"  Model: {model_name}")
    print(f"  Layer: {block_idx if block_idx is not None else 'All layers'}")
    print(f"  Output directory: {output_dir}")
    
    # Dictionary to store results for all models
    all_model_results = {}

    # Track timing
    start_time = time.time()
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test images
        print(f"Loading {num_samples} images from ImageNet...")
        input_tensor = load_imagenet_batch(image_dir, num_samples)
        batch_size = input_tensor.shape[0]
        print(f"Loaded batch of {batch_size} images in {time.time() - start_time:.2f} seconds")

        # Load model
        print(f"Loading model: {model_name}")
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        
        # Get device (use CUDA if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = model.to(device)
        input_tensor = input_tensor.to(device)

        # Run analysis
        print(f"Analyzing model: {model_name}")
        results = run_multilayer_analysis(
            model, 
            input_tensor, 
            target_layer=block_idx, 
            debug=True
        )

        if not results:
            print(f"No results generated for {model_name}. Exiting.")
            return

        # Store results for this model
        all_model_results[model_name] = results

        # Print model-level summary
        if "hierarchical_stats" in results and "model_level" in results["hierarchical_stats"]:
            model_stats = results["hierarchical_stats"]["model_level"]
            print(f"\n===== MODEL-LEVEL SUMMARY FOR {model_name} =====")
            print(f"Number of layers analyzed: {model_stats['num_layers']}")
            print(f"Total heads analyzed: {model_stats['total_heads']}")
            print(f"Average direct similarity: {model_stats['avg_direct_similarity']:.6f} ± {model_stats['std_direct_similarity']:.6f}")
            print(f"Average optimal similarity: {model_stats['avg_optimal_similarity']:.6f} ± {model_stats['std_optimal_similarity']:.6f}")
            print(f"Average improvement: {model_stats['avg_improvement']:.6f}")
            print(f"Average relative Frobenius error: {model_stats['avg_relative_frobenius_error']:.6f}")
            print(f"Average Pearson correlation: {model_stats['avg_pearson_correlation']:.6f}")
            print(f"Average Linear CKA: {model_stats['avg_linear_cka']:.6f}")
            print(f"Average Kernel CKA: {model_stats['avg_kernel_cka']:.6f}")

        # Save statistics for further analysis
        save_hierarchical_statistics(results["hierarchical_stats"], output_dir, model_name)

        # Print allclose statistics
        print(f"\n===== torch.allclose Statistics =====")
        print(f"Passes: {ALLCLOSE_PASS_COUNT}")
        print(f"Total checks: {ALLCLOSE_TOTAL_COUNT}")
        if ALLCLOSE_TOTAL_COUNT > 0:
            print(f"Pass percentage: {(ALLCLOSE_PASS_COUNT / ALLCLOSE_TOTAL_COUNT * 100):.2f}%")
        else:
            print("No allclose checks performed")

        # Save combined results summary
        combined_results = {
            "model_analyzed": model_name,
            "num_samples": num_samples,
            "dataset": image_dir,
            "allclose_statistics": {
                "pass_count": ALLCLOSE_PASS_COUNT,
                "total_count": ALLCLOSE_TOTAL_COUNT,
                "pass_percentage": (
                    (ALLCLOSE_PASS_COUNT / ALLCLOSE_TOTAL_COUNT * 100)
                    if ALLCLOSE_TOTAL_COUNT > 0
                    else 0
                ),
            },
            "model_summary": results["hierarchical_stats"]["model_level"] if "hierarchical_stats" in results and "model_level" in results["hierarchical_stats"] else {}
        }

        # Save to JSON file
        serializable_results = convert_to_json_serializable(combined_results)
        with open(os.path.join(output_dir, f"{model_name}_results_summary.json"), "w") as f:
            json.dump(serializable_results, f, indent=4)

        total_elapsed_time = time.time() - start_time
        print(f"\n===== Analysis completed in {total_elapsed_time:.2f} seconds =====")
        print(f"Results saved to {output_dir}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()