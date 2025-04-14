import torch
import numpy as np
import matplotlib.pyplot as plt
import timm
import seaborn as sns
from tqdm import tqdm
from tueplots import bundles
import matplotlib.pyplot as plt

bundles.icml2024()
plt.rcParams.update({"xtick.labelsize": 38})
plt.rcParams.update({"axes.labelsize": 20})
plt.rcParams.update({"ytick.labelsize": 38})
plt.rcParams.update({"axes.titlesize": 40})
plt.rcParams.update({"legend.fontsize": 25})
plt.rcParams.update({"font.size": 18})
plt.rcParams.update({"legend.title_fontsize": 30})
plt.rcParams.update({"axes.titlepad": 20})  # Set global title padding
plt.rcParams["text.usetex"] = True

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

short_names = {
    "vit_tiny_patch16_224": "ViT-Tiny",
    "vit_small_patch16_224": "ViT-Small",
    "vit_base_patch16_224": "ViT-Base",
    "vit_large_patch16_224": "ViT-Large",
    "deit_small_patch16_224": "DeiT-Small",
    "deit_base_patch16_224": "DeiT-Base",
    "deit_tiny_distilled_patch16_224": "DeiT-Tiny-D",
    "deit_small_distilled_patch16_224": "DeiT-Small-D",
    "deit_base_distilled_patch16_224": "DeiT-Base-D",
    "deit_tiny_patch16_224": "DeiT-Tiny",
}


def analyze_model(model_name):
    print(f"Analyzing model: {model_name}")

    # Create model
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    # Create a dummy input tensor
    # Note: All models in the list expect 224x224 images
    batch_size = 5
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    img_path = "imnet_sample/ILSVRC2012_val_00002349.JPEG"

    # Global list to store attention-weighted values
    attention_outputs = []

    def attn_detailed_hook(module, input, output):
        x = input[0]
        B, N, C = x.shape

        # Handle different attribute names based on model type
        if hasattr(module, "qkv"):
            # Standard ViT models
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, C // module.num_heads
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
        elif hasattr(module, "q") and hasattr(module, "k") and hasattr(module, "v"):
            # Some models separate q, k, v projections
            q = (
                module.q(x)
                .reshape(B, N, module.num_heads, C // module.num_heads)
                .permute(0, 2, 1, 3)
            )
            k = (
                module.k(x)
                .reshape(B, N, module.num_heads, C // module.num_heads)
                .permute(0, 2, 1, 3)
            )
            v = (
                module.v(x)
                .reshape(B, N, module.num_heads, C // module.num_heads)
                .permute(0, 2, 1, 3)
            )
        else:
            print(f"Unsupported attention module structure for {model_name}")
            return

        # Use module.scale if available, otherwise calculate it
        scale = getattr(module, "scale", 1.0 / (C // module.num_heads) ** 0.5)

        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

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

    # Remove any existing hooks
    remove_all_hooks(model)

    # Register hooks to all attention blocks
    hook_handles = []

    # Find attention blocks based on model architecture
    if "vit" in model_name or "deit" in model_name:
        # For ViT and DeiT models
        if hasattr(model, "blocks"):
            for blk in model.blocks:
                if hasattr(blk, "attn"):
                    handle = blk.attn.register_forward_hook(attn_detailed_hook)
                    hook_handles.append(handle)
    else:
        print(f"Unknown model architecture for {model_name}")
        return None

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Get number of layers
    no_layers = len(attention_outputs)
    if no_layers == 0:
        print(f"No attention outputs captured for {model_name}")
        return None

    print(f"Captured {no_layers} attention layers")

    # Get shape information from the first layer
    first_layer = attention_outputs[0]
    num_heads = first_layer["q"].shape[1]
    num_tokens = first_layer["q"].shape[2]

    print(f"Batch size: {batch_size}, Num heads: {num_heads}, Num tokens: {num_tokens}")

    # Data structures to store results
    phi_norms = np.zeros((no_layers, batch_size, num_heads, num_tokens))
    h_norms = np.zeros((no_layers, batch_size, num_heads, num_tokens))

    # Loop through layers, heads, tokens, and batch items
    for layer_idx in range(no_layers):
        x, q, k, v, attn_matr = (
            attention_outputs[layer_idx]["x"],
            attention_outputs[layer_idx]["q"],
            attention_outputs[layer_idx]["k"],
            attention_outputs[layer_idx]["v"],
            attention_outputs[layer_idx]["attn"],
        )

        for h in range(num_heads):
            for i in range(num_tokens):
                for b in range(batch_size):
                    # standardize q and k to have mean 0 and std 1
                    # q_std = q[b, h, i] if q[b, h, i].std() == 0 else (q[b, h, i] - q[b, h, i].mean()) / q[b, h, i].std()
                    # k_std = k[b, h] if k[b, h].std() == 0 else (k[b, h] - k[b, h].mean()) / k[b, h].std()

                    q_std = q[b, h, i]
                    k_std = k[b, h]

                    # Calculate output of attention
                    o = attn_matr @ v

                    # Calculate h norm squared (squared L2 norm of the output vector)
                    h_ = o[b, h, i].pow(2).sum()

                    # Calculate phi norm squared (nom/den)
                    nom = ((q_std * q_std).sum(-1) / 8).exp()
                    den = (((q_std * k_std).sum(-1) / 8).exp().sum()) ** 2

                    # Avoid division by zero
                    if den == 0:
                        phi_norm_squared = 0
                    else:
                        phi_norm_squared = nom / den

                    # Store results
                    phi_norms[layer_idx, b, h, i] = phi_norm_squared
                    h_norms[layer_idx, b, h, i] = h_

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()

    return {
        "model_name": model_name,
        "phi_norms": phi_norms,
        "h_norms": h_norms,
        "no_layers": no_layers,
        "num_heads": num_heads,
        "num_tokens": num_tokens,
        "batch_size": batch_size,
    }


# Replace the LaTeX expressions in the plot_layer_statistics function
# Change \h_i to h_i or \mathbf{h}_i


def plot_layer_statistics(
    model_results, use_percentiles=True, remove_outliers=True, outlier_threshold=95
):
    """
    Plot layer-wise statistics for both metrics

    Parameters:
    - model_results: Dictionary containing model analysis results
    - use_percentiles: If True, use median and quartiles instead of mean and std
    - remove_outliers: If True, remove outliers before calculating statistics
    - outlier_threshold: Percentile threshold for defining outliers
    """
    model_name = model_results["model_name"]
    phi_norms = model_results["phi_norms"]
    h_norms = model_results["h_norms"]
    no_layers = model_results["no_layers"]

    # Create x-axis for layers
    layers = np.arange(no_layers)

    # Arrays to store statistics
    if use_percentiles:
        phi_centers = np.zeros(no_layers)
        phi_lower = np.zeros(no_layers)
        phi_upper = np.zeros(no_layers)
        h_centers = np.zeros(no_layers)
        h_lower = np.zeros(no_layers)
        h_upper = np.zeros(no_layers)
    else:
        phi_means = np.zeros(no_layers)
        phi_stds = np.zeros(no_layers)
        h_means = np.zeros(no_layers)
        h_stds = np.zeros(no_layers)

    # Calculate statistics for each layer
    for layer_idx in range(no_layers):
        phi_data = phi_norms[layer_idx].flatten()
        h_data = h_norms[layer_idx].flatten()

        # Remove outliers if requested
        if remove_outliers:
            phi_threshold = np.percentile(phi_data, outlier_threshold)
            phi_data = phi_data[phi_data <= phi_threshold]

            h_threshold = np.percentile(h_data, outlier_threshold)
            h_data = h_data[h_data <= h_threshold]

        if use_percentiles:
            # Use median and quartiles
            phi_centers[layer_idx] = np.median(phi_data)
            phi_lower[layer_idx] = np.percentile(phi_data, 25)
            phi_upper[layer_idx] = np.percentile(phi_data, 75)

            h_centers[layer_idx] = np.median(h_data)
            h_lower[layer_idx] = np.percentile(h_data, 25)
            h_upper[layer_idx] = np.percentile(h_data, 75)
        else:
            # Use mean and std
            phi_means[layer_idx] = np.mean(phi_data)
            phi_stds[layer_idx] = np.std(phi_data)
            h_means[layer_idx] = np.mean(h_data)
            h_stds[layer_idx] = np.std(h_data)

    # COMBINED PLOTS (REGULAR SCALE) - Both norms on same plot
    """
    plt.figure(figsize=(15, 7))
    
    if use_percentiles:
        # Phi norm (blue)
        plt.fill_between(layers, phi_lower, phi_upper, alpha=0.2, color='blue', label=r'$\left| \varphi(q_i) \right|^2$ Quartiles')
        plt.plot(layers, phi_centers, marker='o', linestyle='-', color='blue', label=r'$\left| \varphi(q_i) \right|^2$ Median')
        
        # H norm (red) - Fixed the LaTeX notation here
        plt.fill_between(layers, h_lower, h_upper, alpha=0.2, color='red', label=r'$\left| \mathbf{h}_i \right|^2$ Quartiles')
        plt.plot(layers, h_centers, marker='s', linestyle='-', color='red', label=r'$\left| \mathbf{h}_i \right|^2$ Median')
    
        plt.ylabel('Norm Squared Value (median)')
    else:
        # Phi norm (blue)
        plt.errorbar(layers, phi_means, yerr=phi_stds, marker='o', linestyle='-', capsize=5, 
                     color='blue', label=r'$\left| \varphi(q_i) \right|^2$ Mean ± Std')
        
        # H norm (red) - Fixed the LaTeX notation here
        plt.errorbar(layers, h_means, yerr=h_stds, marker='s', linestyle='-', capsize=5, 
                     color='red', label=r'$\left| \mathbf{h}_i \right|^2$ Mean ± Std')
        
        plt.ylabel('Norm Squared Value (mean)')
    
    plt.xlabel('Layer Index')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    stats_type = "percentiles" if use_percentiles else "mean/std"
    outlier_info = f"outliers removed (${outlier_threshold}\%$)" if remove_outliers else "with outliers"
    
    plt.tight_layout()
    plt.savefig(f"appendix_a2/{model_name}_combined_stats.pdf")
    plt.show()
    """

    plt.figure(figsize=(18, 9))

    # Add title
    # plt.title("Comparison of Squared Norms over Layers")
    if use_percentiles:
        # Phi norm (blue)
        plt.fill_between(
            layers, phi_lower, phi_upper, alpha=0.2, color="blue"
        )  # label=r'$\left| \varphi(q_i) \right|^2$ Quartiles')
        plt.plot(
            layers, phi_centers, marker="o", linestyle="-", color="blue"
        )  # label=r'$\left| \varphi(q_i) \right|^2$ Median')

        # H norm (red) - Fixed the LaTeX notation here
        plt.fill_between(
            layers, h_lower, h_upper, alpha=0.2, color="red"
        )  # label=r'$\left| \mathbf{h}_i \right|^2$ Quartiles')
        plt.plot(
            layers, h_centers, marker="s", linestyle="-", color="red"
        )  # label=r'$\left| \mathbf{h}_i \right|^2$ Median')

        # put legend to the right of the plot
        # Add the shortename as a text with a light background for better visibilit
        plt.text(
            0.95,  # x-coordinate (more to the right)
            0.05,  # y-coordinate (higher up)
            short_names.get(model_name, model_name),
            fontsize=30,  # Larger font
            ha="right",  # Align text's right edge to x position
            va="bottom",  # Align text's top edge to y position
            transform=plt.gca().transAxes,  # Use axes coordinates (0-1 scale)
            bbox=dict(
                facecolor="white",
                alpha=1.0,  # Fully opaque background
                edgecolor="black",
                boxstyle="round,pad=0.5",  # Rounded corners with more padding
                linewidth=1.5,  # Thicker border
            ),
        )
        # plt.ylabel('Norm Squared Value (median)')
    else:
        # Phi norm (blue)
        plt.errorbar(
            layers,
            phi_means,
            yerr=phi_stds,
            marker="o",
            linestyle="-",
            capsize=5,
            color="blue",
            label=r"$\\| \varphi(q_i) \\|^2$ Mean ± Std",
        )

        # H norm (red) - Fixed the LaTeX notation here
        plt.errorbar(
            layers,
            h_means,
            yerr=h_stds,
            marker="s",
            linestyle="-",
            capsize=5,
            color="red",
            label=r"$\\| \mathbf{h}_i\\|^2$ Mean ± Std",
        )

        # plt.ylabel('Norm Squared Value (mean)')

    # plt.xlabel('Layer Index')
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    plt.xticks(np.arange(0, no_layers, 1))  # Show every layer number

    plt.tight_layout()
    plt.savefig(f"appendix_a2/{model_name}_combined_stats_log.pdf")
    plt.show()

    # Return the statistics for combined plots
    if use_percentiles:
        return {
            "model_name": model_name,
            "layers": layers,
            "phi_centers": phi_centers,
            "phi_lower": phi_lower,
            "phi_upper": phi_upper,
            "h_centers": h_centers,
            "h_lower": h_lower,
            "h_upper": h_upper,
            "stats_type": "percentiles",
        }
    else:
        return {
            "model_name": model_name,
            "layers": layers,
            "phi_means": phi_means,
            "phi_stds": phi_stds,
            "h_means": h_means,
            "h_stds": h_stds,
            "stats_type": "mean/std",
        }


# Also fix the plot_all_models_combined function
def plot_all_models_combined(all_model_stats, use_log_scale=False):
    """
    Plot statistics from all models in one combined plot

    Parameters:
    - all_model_stats: List of dictionaries containing model statistics
    - use_log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(18, 9))
    # plt.title("Comparison of Squared Norms over Layers", fontsize=30)

    # Define a color cycle
    colors = plt.cm.tab10.colors

    for i, stats in enumerate(all_model_stats):
        model_name = stats["model_name"]
        layers = stats["layers"]
        color = colors[i % len(colors)]

        # Shortened model name for legend
        short_name = short_names.get(model_name, model_name)

        if stats["stats_type"] == "percentiles":
            # Plot phi norm with solid line
            plt.plot(
                layers,
                stats["phi_centers"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{short_name} - $\\left|| \\varphi(q_i) \\right||^2$",
            )

            # Plot h norm with dashed line - Fixed the LaTeX notation here
            plt.plot(
                layers,
                stats["h_centers"],
                marker="s",
                linestyle="--",
                color=color,
                label=f"{short_name} - $\\left|| \\mathbf{{h}}_i \\right||^2$",
            )
        else:
            # Plot phi norm with solid line
            plt.plot(
                layers,
                stats["phi_means"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{short_name} - $\\left|| \\varphi(q_i) \\right||^2$",
            )

            # Plot h norm with dashed line - Fixed the LaTeX notation here
            plt.plot(
                layers,
                stats["h_means"],
                marker="s",
                linestyle="--",
                color=color,
                label=f"{short_name} - $\\left|| \\mathbf{{h}}_i \\right||^2$",
            )

    # Set plot title and labels
    scale_type = "Log Scale" if use_log_scale else "Linear Scale"
    stats_type = all_model_stats[0]["stats_type"]
    measure = "Median" if stats_type == "percentiles" else "Mean"
    plt.xlabel("Layer Index (Normalized)")
    # lt.ylabel(f'Norm Squared Value ({measure})')

    if use_log_scale:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    max_layers = max([len(stats["layers"]) for stats in all_model_stats])

    # Set x-ticks to show every layer number (1 by 1)
    plt.xticks(np.arange(0, max_layers, 1))

    plt.tight_layout()
    plt.savefig(f"appendix_a2/all_models_combined{'_log' if use_log_scale else ''}.pdf")
    plt.show()
    """
    Plot statistics from all models in one combined plot
    
    Parameters:
    - all_model_stats: List of dictionaries containing model statistics
    - use_log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(18, 9))

    # Define a color cycle
    colors = plt.cm.tab10.colors

    for i, stats in enumerate(all_model_stats):
        model_name = stats["model_name"]
        layers = stats["layers"]
        color = colors[i % len(colors)]

        # Shortened model name for legend
        short_name = short_names.get(model_name, model_name)

        if stats["stats_type"] == "percentiles":
            # Plot phi norm with solid line
            plt.plot(
                layers,
                stats["phi_centers"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{short_name} - $\\| \\varphi(q_i) \\|^2$",
            )

            # Plot h norm with dashed line
            plt.plot(
                layers,
                stats["h_centers"],
                marker="s",
                linestyle="--",
                color=color,
                label=f"{short_name} - $\\|\\h_i \\|^2$",
            )
        else:
            # Plot phi norm with solid line
            plt.plot(
                layers,
                stats["phi_means"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{short_name} - $\\| \\varphi(q_i) \\|^2$",
            )

            # Plot h norm with dashed line
            plt.plot(
                layers,
                stats["h_means"],
                marker="s",
                linestyle="--",
                color=color,
                label=f"{short_name} - $\\| \\h_i \\|^2$",
            )

    # Set plot title and labels
    scale_type = "Log Scale" if use_log_scale else "Linear Scale"
    stats_type = all_model_stats[0]["stats_type"]
    measure = "Median" if stats_type == "percentiles" else "Mean"

    plt.xlabel("Layer Index (Normalized)")
    # plt.ylabel(f'Norm Squared Value ({measure})')

    if use_log_scale:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f"appendix_a2/all_models_combined{'_log' if use_log_scale else ''}.pdf")
    plt.show()


def compute_layer_statistics(model_results):
    """
    Compute detailed statistics for phi and h norms in each layer

    Parameters:
    - model_results: Dictionary containing model analysis results

    Returns:
    - Dictionary with detailed statistics
    """
    model_name = model_results["model_name"]
    phi_norms = model_results["phi_norms"]
    h_norms = model_results["h_norms"]
    no_layers = model_results["no_layers"]

    # Arrays to store statistics for each layer
    layer_stats = []

    # Track overall statistics across all layers
    all_phi_values = []
    all_h_values = []

    print(f"\n===== Statistics for {model_name} =====")
    print(
        f"{'Layer':<6} | {'Metric':<5} | {'Mean':<15} | {'Median':<15} | {'Min':<15} | {'Max':<15} | {'Ratio (Phi/H)':<15}"
    )
    print("-" * 90)

    # Calculate statistics for each layer
    for layer_idx in range(no_layers):
        phi_data = phi_norms[layer_idx].flatten()
        h_data = h_norms[layer_idx].flatten()

        # Add to overall data
        all_phi_values.extend(phi_data)
        all_h_values.extend(h_data)

        # Calculate statistics
        phi_mean = np.mean(phi_data)
        phi_median = np.median(phi_data)
        phi_min = np.min(phi_data)
        phi_max = np.max(phi_data)

        h_mean = np.mean(h_data)
        h_median = np.median(h_data)
        h_min = np.min(h_data)
        h_max = np.max(h_data)

        # Calculate ratios (handle division by zero)
        mean_ratio = phi_mean / h_mean if h_mean != 0 else float("inf")
        median_ratio = phi_median / h_median if h_median != 0 else float("inf")

        # Store layer statistics
        layer_stats.append(
            {
                "layer": layer_idx,
                "phi_mean": phi_mean,
                "phi_median": phi_median,
                "phi_min": phi_min,
                "phi_max": phi_max,
                "h_mean": h_mean,
                "h_median": h_median,
                "h_min": h_min,
                "h_max": h_max,
                "mean_ratio": mean_ratio,
                "median_ratio": median_ratio,
            }
        )

        # Print phi statistics
        print(
            f"{layer_idx:<6} | {'Phi':<5} | {phi_mean:<15.6e} | {phi_median:<15.6e} | {phi_min:<15.6e} | {phi_max:<15.6e} | {mean_ratio:<15.6e}"
        )

        # Print h statistics
        print(
            f"{'':<6} | {'H':<5} | {h_mean:<15.6e} | {h_median:<15.6e} | {h_min:<15.6e} | {h_max:<15.6e} | {'':<15}"
        )

        # Add separator between layers
        if layer_idx < no_layers - 1:
            print("-" * 90)

    # Calculate overall statistics
    overall_phi_mean = np.mean(all_phi_values)
    overall_phi_median = np.median(all_phi_values)
    overall_phi_min = np.min(all_phi_values)
    overall_phi_max = np.max(all_phi_values)

    overall_h_mean = np.mean(all_h_values)
    overall_h_median = np.median(all_h_values)
    overall_h_min = np.min(all_h_values)
    overall_h_max = np.max(all_h_values)

    # Calculate overall ratio
    overall_mean_ratio = (
        overall_phi_mean / overall_h_mean if overall_h_mean != 0 else float("inf")
    )
    overall_median_ratio = (
        overall_phi_median / overall_h_median if overall_h_median != 0 else float("inf")
    )

    # Print overall statistics
    print("=" * 90)
    print(
        f"{'ALL':<6} | {'Phi':<5} | {overall_phi_mean:<15.6e} | {overall_phi_median:<15.6e} | {overall_phi_min:<15.6e} | {overall_phi_max:<15.6e} | {overall_mean_ratio:<15.6e}"
    )
    print(
        f"{'LAYERS':<6} | {'H':<5} | {overall_h_mean:<15.6e} | {overall_h_median:<15.6e} | {overall_h_min:<15.6e} | {overall_h_max:<15.6e} | {'':<15}"
    )
    print("=" * 90)

    # Return the statistics
    return {
        "model_name": model_name,
        "layer_stats": layer_stats,
        "overall": {
            "phi_mean": overall_phi_mean,
            "phi_median": overall_phi_median,
            "phi_min": overall_phi_min,
            "phi_max": overall_phi_max,
            "h_mean": overall_h_mean,
            "h_median": overall_h_median,
            "h_min": overall_h_min,
            "h_max": overall_h_max,
            "mean_ratio": overall_mean_ratio,
            "median_ratio": overall_median_ratio,
        },
    }


def compute_relative_errors(model_results):
    """
    Compute relative errors between phi and h for visualization

    Parameters:
    - model_results: Dictionary containing model analysis results

    Returns:
    - Dictionary with relative error statistics
    """
    model_name = model_results["model_name"]
    phi_norms = model_results["phi_norms"]
    h_norms = model_results["h_norms"]
    no_layers = model_results["no_layers"]

    # Arrays to store relative errors
    rel_error_wrt_phi = np.zeros(no_layers)  # |phi - h| / |phi|
    rel_error_wrt_h = np.zeros(no_layers)  # |phi - h| / |h|
    abs_error = np.zeros(no_layers)  # |phi - h|

    print(f"\n===== Relative Errors for {model_name} =====")
    print(
        f"{'Layer':<6} | {'Abs Error':<15} | {'Rel Error (wrt Phi)':<25} | {'Rel Error (wrt H)':<25}"
    )
    print("-" * 80)

    # Calculate statistics for each layer
    for layer_idx in range(no_layers):
        phi_data = phi_norms[layer_idx].flatten()
        h_data = h_norms[layer_idx].flatten()

        # Mean values for this layer
        phi_mean = np.mean(phi_data)
        h_mean = np.mean(h_data)

        # Calculate absolute error
        abs_err = np.abs(phi_mean - h_mean)
        abs_error[layer_idx] = abs_err

        # Calculate relative errors (handle division by zero)
        rel_err_phi = abs_err / phi_mean if phi_mean != 0 else float("inf")
        rel_err_h = abs_err / h_mean if h_mean != 0 else float("inf")

        rel_error_wrt_phi[layer_idx] = rel_err_phi
        rel_error_wrt_h[layer_idx] = rel_err_h

        # Print statistics
        print(
            f"{layer_idx:<6} | {abs_err:<15.6e} | {rel_err_phi:<25.6e} | {rel_err_h:<25.6e}"
        )

    # Calculate overall statistics
    mean_abs_error = np.mean(abs_error)
    mean_rel_error_phi = np.mean(rel_error_wrt_phi)
    mean_rel_error_h = np.mean(rel_error_wrt_h)

    # Print overall statistics
    print("=" * 80)
    print(
        f"{'AVG':<6} | {mean_abs_error:<15.6e} | {mean_rel_error_phi:<25.6e} | {mean_rel_error_h:<25.6e}"
    )
    print("=" * 80)

    return {
        "model_name": model_name,
        "abs_error": abs_error,
        "rel_error_wrt_phi": rel_error_wrt_phi,
        "rel_error_wrt_h": rel_error_wrt_h,
        "mean_abs_error": mean_abs_error,
        "mean_rel_error_phi": mean_rel_error_phi,
        "mean_rel_error_h": mean_rel_error_h,
    }


def plot_relative_errors(model_errors, log_scale=True):
    """
    Plot relative errors between phi and h

    Parameters:
    - model_errors: Dictionary containing error statistics
    - log_scale: Whether to use log scale for y-axis
    """
    model_name = model_errors["model_name"]
    no_layers = len(model_errors["abs_error"])
    layers = np.arange(no_layers)

    # Plot relative errors
    plt.figure(figsize=(12, 8))

    plt.plot(
        layers,
        model_errors["rel_error_wrt_phi"],
        marker="o",
        linestyle="-",
        color="blue",
        label=r"$\frac{|\varphi(q_i) - \mathbf{h}_i|}{|\varphi(q_i)|}$",
    )

    plt.plot(
        layers,
        model_errors["rel_error_wrt_h"],
        marker="s",
        linestyle="-",
        color="red",
        label=r"$\frac{|\varphi(q_i) - \mathbf{h}_i|}{|\mathbf{h}_i|}$",
    )

    plt.xlabel("Layer Index")
    # plt.ylabel('Relative Error')
    plt.title(f"Relative Errors for {model_name}")

    if log_scale:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(
        f"appendix_a2/{model_name}_relative_errors{'_log' if log_scale else ''}.pdf"
    )
    plt.show()

    # Plot absolute error
    plt.figure(figsize=(12, 8))

    plt.plot(
        layers,
        model_errors["abs_error"],
        marker="o",
        linestyle="-",
        color="green",
        label=r"$|\varphi(q_i) - \mathbf{h}_i|$",
    )

    plt.xlabel("Layer Index")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error for {model_name}")

    if log_scale:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(
        f"appendix_a2/{model_name}_absolute_error{'_log' if log_scale else ''}.pdf"
    )
    plt.show()


# Add this to the main function
def main():
    # Analyze each model and store results
    all_model_results = []
    all_model_stats = []
    all_model_detailed_stats = []
    all_model_errors = []

    for model_name in SUPPORTED_MODELS:
        try:
            # Analyze model
            results = analyze_model(model_name)

            if results is not None:
                all_model_results.append(results)

                # Generate individual model plots and get statistics
                stats = plot_layer_statistics(
                    results, use_percentiles=True, remove_outliers=True
                )
                all_model_stats.append(stats)

                # Compute detailed statistics
                detailed_stats = compute_layer_statistics(results)
                all_model_detailed_stats.append(detailed_stats)

                # Compute relative errors
                errors = compute_relative_errors(results)
                all_model_errors.append(errors)

                # Plot relative errors
                plot_relative_errors(errors, log_scale=True)

        except Exception as e:
            print(f"Error analyzing model {model_name}: {e}")


if __name__ == "__main__":
    main()
