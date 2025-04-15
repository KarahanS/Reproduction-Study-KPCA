#!/usr/bin/env python
"""
KPCA Gamma Differences Analysis

A standalone script that analyzes gamma differences in transformer models
as described in the KPCA paper (https://arxiv.org/abs/2406.13762).
This extracts just the gamma differences calculation functionality.
"""

import os
import argparse
import logging
import math
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import timm
from torchvision import datasets, transforms
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import yaml
import random


# set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gamma_diff_analysis')

# Function to load config from YAML file
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class GammaDiffAnalyzer:
    """Class for analyzing gamma differences in transformer models."""
    
    def __init__(self, model_name="vit_tiny_patch16_224", block_idx=2, config=None):
        """
        Initialize the gamma differences analyzer.
        
        Args:
            model_name (str): Name of the model from timm
            block_idx (int): Which transformer block to analyze
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.model_name = model_name
        self.block_idx = block_idx
        self.standardize = self.config.get('standardize', False)
        self.random = self.config.get('random', False)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Standardize keys and queries: {self.standardize}")
        logger.info(f"Test with random perturbation: {self.random}")
        
        # Load model
        self.model = self._load_model()
        
        # Set up hooks and prepare for capturing
        self.captured = {}
        self._setup_hooks()
        
        # Get attention parameters
        attn_block = self.model.blocks[self.block_idx].attn
        self.num_heads = attn_block.num_heads
        self.head_dim = attn_block.head_dim
        logger.info(f"Model loaded: {self.model_name} with {self.num_heads} heads and dimension {self.head_dim}")
        
        # Prepare image transformation
        self.transform = self._get_transform()
    
    def _load_model(self):
        """Load the pre-trained model."""
        logger.info(f"Loading model: {self.model_name}")
        model = timm.create_model(self.model_name, pretrained=True)
        model.eval()
        model.to(self.device)
        return model
    
    def _setup_hooks(self):
        """Set up hooks to capture attention values."""
        def att_proj_hook(module, input, output):
            self.captured['att'] = output.detach().clone()
            
        # Register hook on the qkv projection
        attn_block = self.model.blocks[self.block_idx].attn
        attn_block.qkv.register_forward_hook(att_proj_hook)
        logger.info(f"Hooks registered on block {self.block_idx}")
    
    def _get_transform(self):
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_images(self, image_dir, max_images=100):
        """
        Load images from the specified directory.
        
        Args:
            image_dir (str): Directory containing images
            max_images (int): Maximum number of images to load
            
        Returns:
            torch.Tensor: Batch of transformed images
        """
        # Use validation folder for testing
        val_dir = os.path.join(image_dir, 'val')
        if not os.path.exists(val_dir):
            logger.warning(f"Validation directory {val_dir} not found. Using main directory.")
            val_dir = image_dir
            
        # Use ImageFolder to load the data properly
        dataset = datasets.ImageFolder(val_dir, transform=self.transform)
        
        # Sample the dataset
        indices = list(range(len(dataset)))
        if len(indices) > max_images:
            indices = random.sample(indices, max_images)
        else:
            logger.warning(f"Requested {max_images} samples but only found {len(indices)} images.")
            max_images = len(indices)
        
        # Create dataloader with the sampled indices
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=max_images, sampler=sampler)
        
        # Get the batch of images
        for input_tensor, _ in loader:
            logger.info(f"Loaded {input_tensor.shape[0]} images with shape: {input_tensor.shape}")
            break  # Just need one batch
            
        if input_tensor.shape[0] == 0:
            raise ValueError(f"No valid images found in {image_dir}")
            
        return input_tensor.to(self.device)
    
    def forward_pass(self, input_tensor):
        """
        Perform a forward pass through the model.
        
        Args:
            input_tensor (torch.Tensor): Batch of images
            
        Returns:
            tuple: A tuple containing Q, K, V tensors
        """
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Process captured QKV
        qkv = self.captured['att']  # shape: [B, N, 3*D]
        B, N = qkv.shape[0:2]
        
        # Reshape and separate Q, K, V
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)  # each [B, num_heads, N, head_dim]
        
        logger.info(f"Q, K, V shapes: {Q.shape}")
        return Q, K, V
    
    def compute_pairwise_rel_diff(self, gamma_vector):
        """Compute relative pairwise differences between all elements in gamma_vector"""
        N = gamma_vector.shape[0]
        rel_diffs = []
        
        # Compute all pairwise relative differences
        for i in range(N):
            for j in range(i+1, N):
                rel_diff = torch.abs(gamma_vector[i] - gamma_vector[j]) / (torch.max(torch.abs(gamma_vector[i]), torch.abs(gamma_vector[j])) + 1e-10)
                rel_diffs.append(rel_diff)
        
        # Convert to tensor
        rel_diffs = torch.stack(rel_diffs)
        
        # Return mean and standard deviation
        return torch.mean(rel_diffs), torch.std(rel_diffs)
    
    def compute_pairwise_abs_diff(self, gamma_vector):
        """Compute absolute pairwise differences between all elements in gamma_vector"""
        N = gamma_vector.shape[0]
        abs_diffs = []
        
        # Compute all pairwise absolute differences
        for i in range(N):
            for j in range(i+1, N):
                abs_diffs.append(torch.abs(gamma_vector[i] - gamma_vector[j]))
        
        # Convert to tensor
        abs_diffs = torch.stack(abs_diffs)
        
        # Return mean and standard deviation
        return torch.mean(abs_diffs), torch.std(abs_diffs)
    
    def analyze_gamma_diffs(self, Q, K, b=0, h=0):
        """
        Analyze gamma differences in the attention mechanism.
        
        Args:
            Q (torch.Tensor): Query tensor
            K (torch.Tensor): Key tensor
            b (int): Batch index
            h (int): Head index
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing gamma differences for batch {b}, head {h}")
        
        # Extract key for specified batch and head
        K_bh = K[b, h]  # [N, head_dim]
        Q_bh = Q[b, h]  # [N, head_dim]
        N = K_bh.shape[0]
        
        # Standardize if specified in config
        if self.standardize:
            K_bh = (K_bh - K_bh.mean(dim=-1, keepdim=True)) / (K_bh.std(dim=-1, keepdim=True) + 1e-8)
            Q_bh = (Q_bh - Q_bh.mean(dim=-1, keepdim=True)) / (Q_bh.std(dim=-1, keepdim=True) + 1e-8)
        
        # Compute kernel matrix
        K_raw = torch.exp(K_bh @ K_bh.T / self.head_dim**0.5)
        
        # Compute g(x) values
        g_vals = K_raw.sum(dim=1, keepdim=True)  # [N, 1]
        
        # Compute normalized kernel matrix
        K_phi = K_raw / (g_vals @ g_vals.T + 1e-6)  # [N, N]
        
        # Compute centered Gram matrix
        one_N = torch.ones(N, N, device=self.device) / N
        K_centered = K_phi - one_N @ K_phi - K_phi @ one_N + one_N @ K_phi @ one_N
        
        # Compute eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(K_centered)
        sorted_indices = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        
        # Extract top eigenvectors
        A = eigvecs[:, 0:self.head_dim]
        
        # Add small random perturbation and re-orthogonalize if specified in config
        if self.random:
            noise_strength = 0.1
            A_random = A + noise_strength * torch.randn_like(A)
            Q_matrix, _ = torch.linalg.qr(A_random)
            A_random = Q_matrix[:, :self.head_dim]
        else:
            A_random = A  # Use the actual eigenvectors if no randomization
        
        # Calculate gammas
        gammas = (K_centered @ A_random) / (A_random * N)
        
        # Calculate gamma differences
        results = {}
        num_eigenvectors = self.head_dim
        mean_abs_diffs = torch.zeros(num_eigenvectors, device=self.device)
        std_abs_diffs = torch.zeros(num_eigenvectors, device=self.device)
        mean_rel_diffs = torch.zeros(num_eigenvectors, device=self.device)
        std_rel_diffs = torch.zeros(num_eigenvectors, device=self.device)
        gamma_avg = torch.zeros(num_eigenvectors, device=self.device)
        
        for d in range(num_eigenvectors):
            gamma_d = gammas[:, d]  # gamma vector for the d-th eigenvector
            mean_d, std_d = self.compute_pairwise_abs_diff(gamma_d)
            mean_d_relative, std_d_relative = self.compute_pairwise_rel_diff(gamma_d)
            mean_abs_diffs[d] = mean_d
            std_abs_diffs[d] = std_d
            mean_rel_diffs[d] = mean_d_relative
            std_rel_diffs[d] = std_d_relative
            
            avg = torch.mean(gamma_d)
            gamma_avg[d] = avg
        
        results["gamma_means"] = mean_abs_diffs.cpu().numpy()
        results["gamma_stds"] = std_abs_diffs.cpu().numpy()
        results["gamma_rel_means"] = mean_rel_diffs.cpu().numpy()
        results["gamma_rel_stds"] = std_rel_diffs.cpu().numpy()
        results["gamma_averages"] = gamma_avg.cpu().numpy()
        
        return results
    
    def visualize_gamma_diffs(self, all_results, output_dir="./outputs"):
        """
        Create visualizations of gamma differences.
        
        Args:
            all_results (dict): Dictionary of results from analyze_gamma_diffs
            output_dir (str): Output directory for visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect data from all results
        combined_means = []
        combined_stds = []
        
        for key, result in all_results.items():
            if "gamma_means" in result and "gamma_stds" in result:
                combined_means.append(result["gamma_means"])
                combined_stds.append(result["gamma_stds"])
        
        # If no data, return
        if not combined_means:
            logger.warning("No gamma statistics found for visualization")
            return
        
        # Create plot for absolute differences
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate the average across all heads/layers
        all_means = np.vstack(combined_means)
        mean_of_means = np.mean(all_means, axis=0)
        std_of_means = np.std(all_means, axis=0)
        
        # Plot the average with shaded error band
        x = np.arange(mean_of_means.shape[0])
        ax.plot(x, mean_of_means, 'b-', linewidth=1.5, label="Average over attention heads and layers")
        ax.fill_between(x, mean_of_means - std_of_means, mean_of_means + std_of_means, 
                        color='blue', alpha=0.2)
        
        # Add horizontal line at y=0 
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel("Index of the principal component axis")
        ax.set_ylabel(r"$\left| \gamma_i - \gamma_j \right|$")
        ax.set_title(f"Average over attention heads ({self.model_name})")
        
        # Set grid
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.model_name}_gamma_diffs.pdf"), dpi=300)
        plt.close()
        
        # Create plot for relative differences
        combined_rel_means = []
        combined_rel_stds = []
        
        for key, result in all_results.items():
            if "gamma_rel_means" in result and "gamma_rel_stds" in result:
                combined_rel_means.append(result["gamma_rel_means"])
                combined_rel_stds.append(result["gamma_rel_stds"])
        
        if combined_rel_means:
            fig, ax = plt.subplots(figsize=(12, 8))
            all_rel_means = np.vstack(combined_rel_means)
            mean_of_rel_means = np.mean(all_rel_means, axis=0)
            std_of_rel_means = np.std(all_rel_means, axis=0)
            x = np.arange(mean_of_rel_means.shape[0])
            ax.plot(x, mean_of_rel_means, 'g-', linewidth=1.5, label="Average over attention heads and layers")
            ax.fill_between(x, mean_of_rel_means - std_of_rel_means, mean_of_rel_means + std_of_rel_means,
                            color='green', alpha=0.2)
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax.set_xlabel("Index of the principal component axis")
            ax.set_ylabel(r"$ \frac{| \gamma_i - \gamma_j|}{\max(| \gamma_i |, |\gamma_j|)}$")
            ax.set_title(f"Average over attention heads ({self.model_name})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{self.model_name}_gamma_rel_diffs.pdf"), dpi=300)
            plt.close()    
        
        logger.info(f"Gamma differences visualizations saved to {output_dir}")
    
    def run_analysis(self, image_dir, max_images=100, output_dir="./outputs"):
        """
        Run the gamma differences analysis pipeline.
        
        Args:
            image_dir (str): Directory containing images
            max_images (int): Maximum number of images to process
            output_dir (str): Directory to save outputs
                
        Returns:
            dict: Analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load images
        input_tensor = self.load_images(image_dir, max_images)
        
        # 2. Forward pass to get Q, K, V
        Q, K, V = self.forward_pass(input_tensor)
        
        # 3. Determine which samples to analyze
        batch_size = Q.shape[0]
        batch_indices = list(range(min(3, batch_size)))
        
        # 4. Run analysis for all heads
        all_results = {}
        for b in batch_indices:
            for h in range(self.num_heads):
                result_key = f"batch_{b}_head_{h}"
                try:
                    results = self.analyze_gamma_diffs(Q, K, b, h)
                    all_results[result_key] = results
                    logger.info(f"Analysis completed for {result_key}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {result_key}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        # 5. Visualize results
        self.visualize_gamma_diffs(all_results, output_dir)
        
        return all_results


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    image_dir = config.get('dataset_path')
    num_samples = 1
    model_name = config.get('model', 'vit_tiny_patch16_224')
    block_idx = config.get('block_idx', 2)
    seed = config.get('seed', random.randint(0, 10000)) # Default to random seed if not specified
    output_dir = "./gamma_outputs"
    
    # Set random seed
    set_random_seed(seed)
    
    # Set debug logging if requested
    logger.setLevel(logging.INFO)
    
    # Initialize and run analyzer
    try:
        analyzer = GammaDiffAnalyzer(model_name=model_name, block_idx=block_idx, config=config)
        analyzer.run_analysis(
            image_dir=image_dir,
            max_images=num_samples,
            output_dir=output_dir
        )
        logger.info(f"Analysis completed. Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()