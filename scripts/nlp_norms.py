"""
Requirements:
  pip install torch transformers datasets matplotlib tqdm seaborn tueplots pyyaml
"""

import torch, random, os, yaml, math, numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tueplots import bundles

# ---------------------------------------------------------------------
# 1. Global setup & plotting style (unchanged)
# ---------------------------------------------------------------------
bundles.icml2024()
plt.rcParams.update({
    "xtick.labelsize": 38, "ytick.labelsize": 38, "axes.labelsize": 20,
    "axes.titlesize": 40, "legend.fontsize": 25, "legend.title_fontsize": 30,
    "font.size": 18, "axes.titlepad": 20, "text.usetex": False  # disable LaTeX for NLP plots
})


""" from attention paper:
The encoder contains self-attention layers. In a self-attention layer all of the keys, values
and queries come from the same place, in this case, the output of the previous layer in the
encoder. Each position in the encoder can attend to all positions in the previous layer of the
encoder.

The simplest, safest way to avoid the masking complications of decoder‑only LLMs is to restrict your experiment to 
encoder‑only transformers. These models (BERT, RoBERTa, ELECTRA, DistilBERT, ALBERT, DeBERTa, …) apply unmasked 
self‑attention across the whole input, so your φ‑ and h‑norm computations remain exactly in‑sync with the model’s forward 
pass without any extra masking logic. 
They are also smaller than most causal decoders, which keeps GPU memory in check.
"""
SUPPORTED_MODELS = [
    "bert-base-uncased",                 # 12‑layer baseline encoder
    "roberta-base",                      # BERT + training tweaks
    "google/electra-base-discriminator", # discriminator with replaced‑token objective
    "google/electra-small-discriminator",                    # weight‑sharing, factorised embeddings
    "xlm-roberta-base",
    "allenai/longformer-base-4096",
    "sentence-transformers/all-MiniLM-L6-v2",
    "camembert-base",
    "studio-ousia/luke-base",
]

SHORT_NAMES = {
    "bert-base-uncased":                 "BERT‑Base",
    "roberta-base":                      "RoBERTa‑Base",
    "google/electra-base-discriminator": "ELECTRA‑Base",
    "google/electra-small-discriminator": "ELECTRA‑Small",
    "albert-base-v2":                    "ALBERT‑Base",
    "xlm-roberta-base":                  "XLM‑RoBERTa",
    "allenai/longformer-base-4096":      "Longformer",
    "google/bigbird-roberta-base":       "BigBird",
    "funnel-transformer/small":          "Funnel",
    "microsoft/mpnet-base":              "MPNet",
    "google/reformer-enwik8":            "Reformer",
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
    "camembert-base":                    "CamemBERT",
    "studio-ousia/luke-base":            "LUKE",
}



# ---------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic, cudnn.benchmark = True, False

def load_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

# ---------------------------------------------------------------------
# 3. Core analysis function
# ---------------------------------------------------------------------
def analyze_model_nlp(model_name):
    import math                                     # ① missing earlier
    print(f"\n=== Analyzing {model_name} ===")

    # ----------------------------------- load model & tokenizer ------------
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # GPT‑, OPT‑ & friends need an explicit pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ----------------------------------- config ----------------------------
    cfg        = load_config()
    seed = cfg.get('seed', random.randint(0, 10000))  # Default to random seed if not specified
    sentences  = cfg.get("sentences") or [
        "Transformers are changing natural language processing.",
        "PyTorch hooks let us peek inside neural networks."
    ]
    set_random_seed(seed)

    enc = tokenizer(
        sentences,
        padding   = True,
        truncation= True,
        max_length= 128,
        return_tensors="pt"
    ).to(device)

    batch_size = enc["input_ids"].size(0)

    # ----------------------------------- storage ---------------------------
    attention_outputs = []

    # ----------------------------------- hook ------------------------------
    def attn_hook(module, inp, out):
        hidden = inp[0]                       # (B, L, H)
        B, L, Hdim = hidden.shape
        
        print(f"Layer {len(attention_outputs)}: {module.__class__.__name__} ({B}, {L}, {Hdim})")

        # -------- identify projections (works for old & new classes) -------
        if all(hasattr(module, x) for x in ("query", "key", "value")):
            print("Using query, key, value")
            q_proj, k_proj, v_proj = module.query, module.key, module.value
            if hasattr(module, "num_attention_heads"):
                # BART/OPT style
                head_dim  = module.attention_head_size
                num_heads = module.num_attention_heads
            elif hasattr(module, "num_heads"):
                # GPT‑2 style
                head_dim  = module.head_dim
                num_heads = module.num_heads
         

            q = q_proj(hidden)
            k = k_proj(hidden)
            v = v_proj(hidden)

        elif all(hasattr(module, x) for x in ("q_proj", "k_proj", "v_proj")):
            print("Using q_proj, k_proj, v_proj")   
            q_proj, k_proj, v_proj = module.q_proj, module.k_proj, module.v_proj
            # Bart/OPT style
            head_dim  = module.head_dim
            num_heads = module.num_heads

            q = q_proj(hidden)
            k = k_proj(hidden)
            v = v_proj(hidden)

        elif all(hasattr(module, x) for x in ("q_lin", "k_lin", "v_lin")):
            print("Using q_lin, k_lin, v_lin")
            q_proj, k_proj, v_proj = module.q_lin, module.k_lin, module.v_lin
            head_dim  = module.dim // module.n_heads
            num_heads = module.n_heads

            q = q_proj(hidden)
            k = k_proj(hidden)
            v = v_proj(hidden)

        elif hasattr(module, "c_attn"):                     # GPT‑2
            num_heads = module.num_heads 
            head_dim  = Hdim // num_heads
            proj = module.c_attn(hidden).split(Hdim, dim=2) # (q,k,v)
            q, k, v = proj

        else:
            # Unknown attention variant – skip
            return

        # reshape to (B, heads, L, head_dim)
        if hasattr(module, "c_attn"):
            q = q.view(B, L, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, L, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, L, num_heads, head_dim).transpose(1, 2)
        else:
            q = q.view(B, L, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, L, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, L, num_heads, head_dim).transpose(1, 2)

        # -------- scaled‑dot‑product attention -----------------------------
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # ---------- ADD THIS ----------
        if model.config.is_decoder and attn_scores.size(-2) == attn_scores.size(-1):
            seq_len = attn_scores.size(-1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len,
                                                device=attn_scores.device))
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        # GPT‑Neo local‑attention patch (optional):
        if hasattr(model.config, "attention_types"):
            # every other layer uses local window size 256
            # implement window mask here if needed
            pass
        # --------------------------------

        attn_probs = attn_scores.softmax(dim=-1)
        o = torch.matmul(attn_probs, v)

        attention_outputs.append({
            "q":     q.detach().cpu(),
            "k":     k.detach().cpu(),
            "v":     v.detach().cpu(),
            "attn":  attn_probs.detach().cpu(),
            "o":     o.detach().cpu(),
        })
        
        print("o shape:", o.shape) # (B, heads, L, head_dim)  - L: sequence length

    # ----------------------------------- register hooks --------------------
    hook_handles = []
    for name, mod in model.named_modules():
        # pick *any* module that looks like self‑attention
        if any(hasattr(mod, attr) for attr in
               ("query", "q_proj", "q_lin", "c_attn")):

            hook_handles.append(mod.register_forward_hook(attn_hook))
    # ----------------------------------- forward pass ----------------------
    with torch.no_grad():
        model(**enc)

    for h in hook_handles:
        h.remove()
        
    print(f"→ removed {len(hook_handles)} attention hooks")

    if not attention_outputs:
        print("⚠️  No attention captured – model layout unknown.")
        return None

    # ----------------------------------- norms -----------------------------
    n_layers = len(attention_outputs)
    n_heads  = attention_outputs[0]["q"].shape[1]
    n_tokens = attention_outputs[0]["q"].shape[2]


    phi_norms = np.zeros((n_layers, batch_size, n_heads, n_tokens))
    h_norms   = np.zeros((n_layers, batch_size, n_heads, n_tokens))

    for l in range(n_layers):
        q = attention_outputs[l]["q"].numpy()
        k = attention_outputs[l]["k"].numpy()
        v = attention_outputs[l]["v"].numpy()
        attn = attention_outputs[l]["attn"].numpy()
        o = attention_outputs[l]["o"].numpy()
        # (B, heads, L, head_dim)  - L: sequence length


        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(n_tokens):
                    q_std = q[b, h, i]
                    k_std = k[b, h]
                    h_sq  = (o[b, h, i] ** 2).sum()
                            
                    log_nom = (q_std * q_std).sum() / 8
                    exp_values = np.exp((q_std * k_std).sum(axis=-1) / 8)
                    log_den = 2 * np.log(np.sum(exp_values))  # since den = (sum(...))²
                    phi_sq = 0 if np.isinf(log_den) else np.exp(log_nom - log_den) 

                    phi_norms[l, b, h, i] = phi_sq
                    h_norms[l, b, h, i]   = h_sq

    return {
        "model_name":  model_name,
        "phi_norms":   phi_norms,
        "h_norms":     h_norms,
        "no_layers":   n_layers,
        "num_heads":   n_heads,
        "num_tokens":  n_tokens,
        "batch_size":  batch_size,
    }

# ---------------------------------------------------------------------
# 4. Plotting & statistics (identical to your original functions)
# ---------------------------------------------------------------------


def plot_layer_statistics(
    model_results, use_percentiles=True, remove_outliers=True, outlier_threshold=95):
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
        if remove_outliers and len(phi_data) > 0:
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
            SHORT_NAMES.get(model_name, model_name),
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
    # replace "/" with "_" in model_name
    model_name = model_name.replace("/", "_")
    plt.savefig(f"nlp_norm_outputs/{model_name}_combined_stats_log.pdf")
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
    batch_size = model_results["batch_size"]

    # Arrays to store statistics for each layer
    layer_stats = []

    # Track overall statistics across all layers
    all_phi_values = []
    all_h_values = []

    print(f"\n===== Statistics for {model_name} (Averaged over {batch_size} images) =====")
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
    batch_size = model_results["batch_size"]

    # Arrays to store relative errors
    rel_error_wrt_phi = np.zeros(no_layers)  # |phi - h| / |phi|
    rel_error_wrt_h = np.zeros(no_layers)  # |phi - h| / |h|
    abs_error = np.zeros(no_layers)  # |phi - h|

    print(f"\n===== Relative Errors for {model_name} (Averaged over {batch_size} images) =====")
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
    # replace "/" with "_" in model_name
    model_name = model_name.replace("/", "_")
    plt.savefig(
        f"nlp_norm_outputs/{model_name}_relative_errors{'_log' if log_scale else ''}.pdf"
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
    # replace "/" with "_" in model_name
    model_name = model_name.replace("/", "_")
    plt.savefig(
        f"nlp_norm_outputs/{model_name}_absolute_error{'_log' if log_scale else ''}.pdf"
    )
    plt.show()


# ---------------------------------------------------------------------
# 5. Main driver
# ---------------------------------------------------------------------
def main():
    if not os.path.exists("nlp_norm_outputs"):
        os.makedirs("nlp_norm_outputs")

    for model_name in SUPPORTED_MODELS:
        try:
            res = analyze_model_nlp(model_name)
            if res is None:
                continue
            # Call your existing utilities
            stats  = plot_layer_statistics(res, use_percentiles=True)
            _      = compute_layer_statistics(res)
            errs   = compute_relative_errors(res)
            _      = plot_relative_errors(errs)
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")

if __name__ == "__main__":
    main()
