import numpy as np
import matplotlib.pyplot as plt
import os
from tueplots import bundles
import pandas as pd
from matplotlib.gridspec import GridSpec

# Set up plot styling with tueplots
plt.rcParams.update(bundles.icml2024(column="half", nrows=1, ncols=1, usetex=True))
plt.rcParams.update({"xtick.labelsize": 40})
plt.rcParams.update({"axes.labelsize": 40})
plt.rcParams.update({"ytick.labelsize": 40})
plt.rcParams.update({"axes.titlesize": 40})
plt.rcParams.update({"legend.fontsize": 40})
plt.rcParams.update({"font.size": 40})
plt.rcParams.update({"legend.title_fontsize": 40})
plt.rcParams.update({"axes.titlepad": 20})  # Set global title padding
plt.rcParams["text.usetex"] = True

# Configuration
# Base folder
data_root = "wandb_data"

# Epochs to start plotting from
phiq_start_epoch = 0
h_start_epoch = 10

# File paths mapping
file_paths = {
    "DeiT-Tiny": {
        "train_rel_error_wrt_phiq": f"{data_root}/deit_tiny/deit_train_rel_error_wrt_varphi.csv",
        "test_rel_error_wrt_phiq": f"{data_root}/deit_tiny/deit_test_rel_error_wrt_varphi.csv",
        "train_rel_error_wrt_h": f"{data_root}/deit_tiny/deit_train_rel_error_wrt_h.csv",
        "test_rel_error_wrt_h": f"{data_root}/deit_tiny/deit_test_rel_error_wrt_h.csv"
    },
    "ViT-Tiny": {
        "train_rel_error_wrt_phiq": f"{data_root}/vit_tiny/vit_train_rel_error_wrt_varphi.csv",
        "test_rel_error_wrt_phiq": f"{data_root}/vit_tiny/vit_test_rel_error_wrt_varphi.csv",
        "train_rel_error_wrt_h": f"{data_root}/vit_tiny/vit_train_rel_error_wrt_h.csv",
        "test_rel_error_wrt_h": f"{data_root}/vit_tiny/vit_test_rel_error_wrt_h.csv"
    }
}

# Colors for plots
train_color = "#0066CC"  # Blue for train
test_color = "#CC0000"   # Red for test

# Create output directory
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

def read_csv_file(file_path):
    """Read a CSV file with W&B-style column names."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Print column names for debugging
        print(f"Columns in {os.path.basename(file_path)}:")
        for col in df.columns:
            print(f"  - {col}")
            
        # Return the original dataframe - we'll handle column selection in the plotting function
        return df
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame(columns=['Step'])

def get_plot_data(model_name, error_type):
    """Get data for plotting without creating the actual plot."""
    # Set appropriate file paths and parameters based on error type
    if error_type == "phiq":
        train_path = file_paths[model_name]["train_rel_error_wrt_phiq"]
        test_path = file_paths[model_name]["test_rel_error_wrt_phiq"]
        start_epoch = phiq_start_epoch
        # Column pattern to look for
        pattern = "rel_error_wrt_phiq"
    else:  # h
        train_path = file_paths[model_name]["train_rel_error_wrt_h"]
        test_path = file_paths[model_name]["test_rel_error_wrt_h"]
        start_epoch = h_start_epoch
        # Column pattern to look for
        pattern = "rel_error_wrt_h"
    
    print(f"\n===== Processing {model_name} relative error data for {error_type} =====")
    
    # Read train and test data
    print(f"Reading train data from: {train_path}")
    train_df = read_csv_file(train_path)
    
    print(f"Reading test data from: {test_path}")
    test_df = read_csv_file(test_path)
    
    # Check if we have data
    if train_df.empty and test_df.empty:
        print("No data found.")
        return None, None, 1, ""
    
    # Find the appropriate columns for plotting
    train_value_col = None
    if not train_df.empty:
        for col in train_df.columns:
            if pattern in col.lower() and not (col.endswith('_MIN') or col.endswith('_MAX')):
                train_value_col = col
                break
        
        if train_value_col:
            print(f"Using train column: {train_value_col}")
        else:
            print("Could not find appropriate train value column")
            train_df = pd.DataFrame(columns=['Step', 'Value', 'epoch'])
    
    test_value_col = None
    if not test_df.empty:
        for col in test_df.columns:
            if pattern in col.lower() and not (col.endswith('_MIN') or col.endswith('_MAX')):
                test_value_col = col
                break
                
        if test_value_col:
            print(f"Using test column: {test_value_col}")
        else:
            print("Could not find appropriate test value column")
            test_df = pd.DataFrame(columns=['Step', 'Value', 'epoch'])
    
    # Create new dataframes with the correct columns
    if not train_df.empty and train_value_col:
        train_plot_df = pd.DataFrame({
            'Step': train_df['Step'],
            'Value': train_df[train_value_col],
            'epoch': train_df['Step'] / 15
        })
        # Filter by start epoch and clean data
        train_plot_df = train_plot_df[train_plot_df['epoch'] >= start_epoch].dropna(subset=['Value']).sort_values(by='epoch')
        print(f"Train data: {len(train_plot_df)} points after filtering")
    else:
        train_plot_df = pd.DataFrame(columns=['Step', 'Value', 'epoch'])
    
    if not test_df.empty and test_value_col:
        test_plot_df = pd.DataFrame({
            'Step': test_df['Step'],
            'Value': test_df[test_value_col],
            'epoch': test_df['Step'] / 15
        })
        # Filter by start epoch and clean data
        test_plot_df = test_plot_df[test_plot_df['epoch'] >= start_epoch].dropna(subset=['Value']).sort_values(by='epoch')
        print(f"Test data: {len(test_plot_df)} points after filtering")
    else:
        test_plot_df = pd.DataFrame(columns=['Step', 'Value', 'epoch'])
    
    # Determine if we need scaling (for very large or small numbers)
    scale_factor = 1
    scale_text = ""
    
    # Collect all values to calculate the appropriate scale
    all_values = []
    if not train_plot_df.empty:
        all_values.extend(train_plot_df['Value'].tolist())
    if not test_plot_df.empty:
        all_values.extend(test_plot_df['Value'].tolist())
    
    if all_values:
        max_value = max([abs(v) for v in all_values if not np.isnan(v)])
        
        # Set scale based on the magnitude
        if max_value >= 1_000_000:
            scale_factor = 1_000_000
            scale_text = r" \times 10^{-6}"
        elif max_value >= 1_000:
            scale_factor = 1_000
            scale_text = r" \times 10^{-3}"
    
    # Apply scaling if needed
    if scale_factor > 1:
        if not train_plot_df.empty:
            train_plot_df['scaled_value'] = train_plot_df['Value'] / scale_factor
        if not test_plot_df.empty:
            test_plot_df['scaled_value'] = test_plot_df['Value'] / scale_factor
    else:
        if not train_plot_df.empty:
            train_plot_df['scaled_value'] = train_plot_df['Value']
        if not test_plot_df.empty:
            test_plot_df['scaled_value'] = test_plot_df['Value']
    
    return train_plot_df, test_plot_df, scale_factor, scale_text

def create_combined_plot(model_name="DeiT-Tiny"):
    """Create a combined plot with both h and phi(q) relative errors."""
    
    # Get data for both types of relative error
    train_h_df, test_h_df, h_scale_factor, h_scale_text = get_plot_data(model_name, "h")
    train_phiq_df, test_phiq_df, phiq_scale_factor, phiq_scale_text = get_plot_data(model_name, "phiq")
    
    # Calculate overall x-axis limits
    min_epoch = 0
    max_epoch = 140  # Default max epoch
    
    for df in [train_h_df, test_h_df, train_phiq_df, test_phiq_df]:
        if df is not None and not df.empty:
            max_epoch = max(max_epoch, df['epoch'].max())
    
    # Create figure with two subplots sharing the x-axis
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.25)  # Increased hspace for more room between plots
    
    # Top plot for h-relative error
    ax1 = fig.add_subplot(gs[0])
    
    # Check if we have h-relative error data
    if train_h_df is not None and not train_h_df.empty:
        ax1.plot(
            train_h_df['epoch'],
            train_h_df['scaled_value'],
            color=train_color,
            linestyle='-',
            linewidth=3,
            label='Train'
        )
    
    if test_h_df is not None and not test_h_df.empty:
        ax1.plot(
            test_h_df['epoch'],
            test_h_df['scaled_value'],
            color=test_color,
            linestyle='-',
            linewidth=3,
            label='Test'
        )
    
    # Set title and y-label for top plot - moved to top position
    ax1.set_title(f"{model_name} Relative Error wrt. $\\| \\mathbf{{h}}_i \\|^2$", pad=20, fontsize=50)
    
    # FIX: Properly formatted LaTeX for h-ylabel 
    h_ylabel = r'$\frac{\left| \|\mathbf{h}_i\|^2 - \|\varphi(\mathbf{q}_i)\|^2 \right|}{\|\mathbf{h}_i\|^2}$'
    
    # Add scale text if needed
    if h_scale_text:
        h_ylabel = h_ylabel[:-1] + h_scale_text + "$"
        
    ax1.set_ylabel(h_ylabel, fontsize=40)
    
    # Set y-limits for h-plot (typically around 1.0)
    ax1.set_ylim(0.9, 1.1)
    ax1.set_yticks(np.arange(0.9, 1.11, 0.05))
    
    # Add legend to top plot
    ax1.legend(fontsize=40, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Hide x-ticks for top plot
    ax1.set_xticklabels([])
    
    # Bottom plot for phi(q)-relative error
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Check if we have phi(q)-relative error data
    if train_phiq_df is not None and not train_phiq_df.empty:
        ax2.plot(
            train_phiq_df['epoch'],
            train_phiq_df['scaled_value'],
            color=train_color,
            linestyle='-',
            linewidth=3,
            label='Train'
        )
    
    if test_phiq_df is not None and not test_phiq_df.empty:
        ax2.plot(
            test_phiq_df['epoch'],
            test_phiq_df['scaled_value'],
            color=test_color,
            linestyle='-',
            linewidth=3,
            label='Test'
        )
    
    # Set title and y-label for bottom plot with increased padding
    ax2.set_title(f"{model_name} Relative Error wrt. $\\| \\varphi(\\mathbf{{q}}_i) \\|^2$", pad=20, fontsize=50)
    
    # FIX: Properly formatted LaTeX for phiq_ylabel
    phiq_ylabel = r'$\frac{\left| \|\mathbf{h}_i\|^2 - \|\varphi(\mathbf{q}_i)\|^2 \right|}{\|\varphi(\mathbf{q}_i)\|^2}$'
    
    # Add scale text if needed
    if phiq_scale_text:
        phiq_ylabel = phiq_ylabel[:-1] + phiq_scale_text + "$"
        
    ax2.set_ylabel(phiq_ylabel, fontsize=40)
    
    # Add legend to bottom plot
    ax2.legend(fontsize=40, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Set x-limits and ticks
    ax1.set_xlim(min_epoch, max_epoch)
    x_ticks = np.arange(0, max_epoch + 10, 10)
    ax2.set_xticks(x_ticks)
    ax2.set_xlabel('Epochs', fontsize=40)
    
    # Tight layout to ensure everything fits nicely
    plt.tight_layout()
    
    # Save the combined plot
    output_path = os.path.join(output_dir, f"{model_name.lower().replace('-', '_')}_combined_relative_errors.pdf")
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined plot to {output_path}")
    
    # Close the figure
    plt.close()

# Create combined plots for each model
try:
    # 1. DeiT-Tiny combined plot
    create_combined_plot("DeiT-Tiny")
    
    # 2. ViT-Tiny combined plot
    create_combined_plot("ViT-Tiny")
    
    print("\nAll combined plots created successfully!")
except Exception as e:
    print(f"\nError creating combined plots: {e}")
    import traceback
    traceback.print_exc()