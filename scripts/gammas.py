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
from torchvision import transforms
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gamma_diff_analysis')

class GammaDiffAnalyzer:
    """Class for analyzing gamma differences in transformer models."""
    
    def __init__(self, model_name="vit_tiny_patch16_224", block_idx=2):
        """
        Initialize the gamma differences analyzer.
        
        Args:
            model_name (str): Name of the model from timm
            block_idx (int): Which transformer block to analyze
        """
        self.model_name = model_name
        self.block_idx = block_idx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
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
        os.makedirs(image_dir, exist_ok=True)
        image_patterns = ['*.JPEG', '*.jpg', '*.png']
        
        # Find all images matching the patterns
        image_paths = []
        for pattern in image_patterns:
            image_paths.extend(glob(os.path.join(image_dir, pattern)))
        
        image_paths = sorted(image_paths)
        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        
        if max_images:
            image_paths = image_paths[:max_images]
            logger.info(f"Using first {len(image_paths)} images")
        
        # Load and transform images
        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert('RGB')
                images.append(self.transform(img))
            except Exception as e:
                logger.warning(f"Error loading image {p}: {e}")
        
        if not images:
            raise ValueError(f"No valid images found in {image_dir}")
            
        input_tensor = torch.stack(images).to(self.device)
        logger.info(f"Loaded image batch with shape: {input_tensor.shape}")
        return input_tensor
    
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
        N = K_bh.shape[0]
        
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
        
        # Add small random perturbation and re-orthogonalize
        noise_strength = 0.1
        A_random = A + noise_strength * torch.randn_like(A)
        Q, _ = torch.linalg.qr(A_random)
        A_random = Q[:, :self.head_dim]
        
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
        ax.set_title("Average over attention heads and layers")
        
        # Set grid
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gamma_diffs.png"), dpi=300)
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
            ax.set_title("Average over attention heads and layers")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "gamma_rel_diffs.png"), dpi=300)
            plt.close()
        
        # Create plot for gamma averages
        combined_avg = []
        for key, result in all_results.items():
            if "gamma_averages" in result:
                combined_avg.append(result["gamma_averages"])
        
        if combined_avg:
            fig, ax = plt.subplots(figsize=(12, 8))
            all_avg = np.vstack(combined_avg)
            mean_of_avg = np.mean(all_avg, axis=0)
            std_of_avg = np.std(all_avg, axis=0)
            x = np.arange(mean_of_avg.shape[0])
            ax.plot(x, mean_of_avg, 'y-', linewidth=1.5, label="Average over attention heads and layers")
            ax.fill_between(x, mean_of_avg - std_of_avg, mean_of_avg + std_of_avg,
                            color='yellow', alpha=0.2)
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax.set_xlabel("Index of the principal component axis")
            ax.set_ylabel("γᵢ")
            ax.set_title("Average over attention heads and layers")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "gamma_avgs.png"), dpi=300)
            plt.close()
        
        logger.info(f"Gamma differences visualizations saved to {output_dir}")
    
    def run_analysis(self, image_dir, max_images=100, output_dir="./outputs", 
                     batch_indices=None, head_indices=None):
        """
        Run the gamma differences analysis pipeline.
        
        Args:
            image_dir (str): Directory containing images
            max_images (int): Maximum number of images to process
            output_dir (str): Directory to save outputs
            batch_indices (list): Which batch indices to analyze
            head_indices (list): Which head indices to analyze
                
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
        if batch_indices is None:
            batch_indices = [0, 1, 2][:min(3, batch_size)]
        
        if head_indices is None:
            head_indices = [0, 1, 2][:min(3, self.num_heads)]
        
        # 4. Run analysis for specified batches and heads
        all_results = {}
        for b in batch_indices:
            if b >= batch_size:
                logger.warning(f"Batch index {b} out of range, skipping")
                continue
                    
            for h in head_indices:
                if h >= self.num_heads:
                    logger.warning(f"Head index {h} out of range, skipping")
                    continue
                
                result_key = f"batch_{b}_head_{h}"
                try:
                    results = self.analyze_gamma_diffs(Q, K, b, h)
                    all_results[result_key] = results
                    logger.info(f"Analysis completed for {result_key}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {result_key}: {e}")
        
        # 5. Visualize results
        self.visualize_gamma_diffs(all_results, output_dir)
        
        return all_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="KPCA Gamma Differences Analysis")
    parser.add_argument("--image_dir", type=str,default="images", help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--model", type=str, default="vit_tiny_patch16_224", help="Model name from timm")
    parser.add_argument("--block_idx", type=int, default=2, help="Transformer block index to analyze")
    parser.add_argument("--max_images", type=int, default=100, help="Maximum number of images to process")
    parser.add_argument("--batch_indices", type=str, default="0,1,2", help="Comma-separated batch indices to analyze")
    parser.add_argument("--head_indices", type=str, default="0,1,2", help="Comma-separated head indices to analyze")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Parse batch and head indices
    batch_indices = [int(i) for i in args.batch_indices.split(',')]
    head_indices = [int(i) for i in args.head_indices.split(',')]
    
    # Initialize and run analyzer
    try:
        analyzer = GammaDiffAnalyzer(model_name=args.model, block_idx=args.block_idx)
        analyzer.run_analysis(
            image_dir=args.image_dir,
            max_images=args.max_images,
            output_dir=args.output_dir,
            batch_indices=batch_indices,
            head_indices=head_indices
        )
        logger.info(f"Analysis completed. Results saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()