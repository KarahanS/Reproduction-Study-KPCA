import numpy as np
import matplotlib.pyplot as plt
import os
from tueplots import bundles
import pandas as pd

# Set up plot styling with tueplots
plt.rcParams.update(bundles.icml2024(column="half", nrows=1, ncols=1, usetex=True))
plt.rcParams.update({"xtick.labelsize": 40})
plt.rcParams.update({"axes.labelsize": 40})
plt.rcParams.update({"ytick.labelsize": 40})
plt.rcParams.update({"axes.titlesize": 40})
plt.rcParams.update({"legend.fontsize": 40})
plt.rcParams.update({"font.size": 45})
plt.rcParams.update({"legend.title_fontsize": 40})
plt.rcParams.update({"axes.titlepad": 25})  # Set global title padding
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
output_dir = "reconstruction_outputs"
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

def plot_relative_error(model_name, error_type):
    """Create a relative error plot for the specified model and error type."""
    # Set appropriate file paths and parameters based on error type
    if error_type == "phiq":
        train_path = file_paths[model_name]["train_rel_error_wrt_phiq"]
        test_path = file_paths[model_name]["test_rel_error_wrt_phiq"]
        title = f"{model_name} Relative Error wrt. $\\| \\varphi(\\mathbf{{q}}_i) \\|^2$"
        ylabel = r'$\frac{\left| \|\mathbf{h}_i\|^2 - \|\varphi(\mathbf{q}_i)\|^2 \right|}{\|\varphi(\mathbf{q}_i)\|^2}$'
        start_epoch = phiq_start_epoch
        output_file = f"{model_name.lower().replace('-', '_')}_relative_error_wrt_phiq.pdf"
        # Column pattern to look for
        pattern = "rel_error_wrt_phiq"
    else:  # h
        train_path = file_paths[model_name]["train_rel_error_wrt_h"]
        test_path = file_paths[model_name]["test_rel_error_wrt_h"]
        title = f"{model_name} Relative Error wrt. $\\| \\mathbf{{h}}_i \\|^2$"
        ylabel = r'$\frac{\left| \|\mathbf{h}_i\|^2 - \|\varphi(\mathbf{q}_i)\|^2 \right|}{\|\mathbf{h}_i\|^2}$'
        start_epoch = h_start_epoch
        output_file = f"{model_name.lower().replace('-', '_')}_relative_error_wrt_h.pdf"
        # Column pattern to look for
        pattern = "rel_error_wrt_h"
    
    print(f"\n===== Creating {model_name} relative error plot for {error_type} =====")
    
    # Create new figure
    plt.figure(figsize=(15, 9))
    
    # Read train and test data
    print(f"Reading train data from: {train_path}")
    train_df = read_csv_file(train_path)
    
    print(f"Reading test data from: {test_path}")
    test_df = read_csv_file(test_path)
    
    # Check if we have data
    if train_df.empty and test_df.empty:
        print("No data found. Skipping plot.")
        plt.close()
        return
    
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
        # Update the y-label with the scale
        ylabel = ylabel[:-1] + scale_text + '$'
    else:
        if not train_plot_df.empty:
            train_plot_df['scaled_value'] = train_plot_df['Value']
        if not test_plot_df.empty:
            test_plot_df['scaled_value'] = test_plot_df['Value']
    
    # Plot train data
    if not train_plot_df.empty:
        plt.plot(
            train_plot_df['epoch'],
            train_plot_df['scaled_value'],
            color=train_color,
            linestyle='-',
            linewidth=2,
            label='Train Relative Error'
        )
        print(f"Plotted train data: range {train_plot_df['scaled_value'].min():.4f} to {train_plot_df['scaled_value'].max():.4f}")
    
    # Plot test data
    if not test_plot_df.empty:
        plt.plot(
            test_plot_df['epoch'],
            test_plot_df['scaled_value'],
            color=test_color,
            linestyle='-',
            linewidth=2,
            label='Test Relative Error'
        )
        print(f"Plotted test data: range {test_plot_df['scaled_value'].min():.4f} to {test_plot_df['scaled_value'].max():.4f}")
    
    # Calculate y-axis limits
    y_values = []
    if not train_plot_df.empty:
        y_values.extend(train_plot_df['scaled_value'].dropna().tolist())
    if not test_plot_df.empty:
        y_values.extend(test_plot_df['scaled_value'].dropna().tolist())
    
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        
        # Special case for h-type errors which tend to hover around 1.0
        if error_type == "h" and 0.8 < np.mean(y_values) < 1.2:
            plt.ylim(0.9, 1.1)
            plt.yticks(np.arange(0.9, 1.11, 0.05))
        else:
            # Add some padding to the y-limits
            padding = (y_max - y_min) * 0.1
            y_bottom = max(0, y_min - padding)  # Don't go below 0 if data is all positive
            y_top = y_max + padding
            
            plt.ylim(y_bottom, y_top)
            
            # Determine appropriate y-tick step
            y_range = y_top - y_bottom
            if y_range <= 0.2:
                y_step = 0.05
            elif y_range <= 1:
                y_step = 0.2
            elif y_range <= 5:
                y_step = 1
            else:
                y_step = np.ceil(y_range / 5)  # Aim for about 5 tick marks
                
            plt.yticks(np.arange(y_bottom, y_top + y_step, y_step))
    
    # Calculate x-axis limits and ticks
    max_epoch = 0
    if not train_plot_df.empty:
        max_epoch = max(max_epoch, train_plot_df['epoch'].max())
    if not test_plot_df.empty:
        max_epoch = max(max_epoch, test_plot_df['epoch'].max())
    
    x_ticks = np.arange(start_epoch, max_epoch + 10, 15)
    plt.xticks(x_ticks)
    
    # Set plot labels and title
    plt.xlabel('Epochs')
    plt.title(title)
    plt.ylabel(ylabel)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=train_color, lw=2),  # Blue for train
        Line2D([0], [0], color=test_color, lw=2),   # Red for test
    ]
    custom_labels = ['Train', 'Test']
    
    plt.legend(
        custom_lines, 
        custom_labels, 
        loc='upper right',
        fontsize=40,
        framealpha=0.9
    )
    
    # Ensure everything fits nicely
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    
    # Close the figure
    plt.close()

# Create all four plots with proper error handling
try:
    # 1. DeiT-Tiny relative error wrt φ(q)
    plot_relative_error("DeiT-Tiny", "phiq")

    # 2. DeiT-Tiny relative error wrt h
    plot_relative_error("DeiT-Tiny", "h")

    # 3. ViT-Tiny relative error wrt φ(q)
    plot_relative_error("ViT-Tiny", "phiq")

    # 4. ViT-Tiny relative error wrt h
    plot_relative_error("ViT-Tiny", "h")

    print("\nAll plots created successfully!")
except Exception as e:
    print(f"\nError creating plots: {e}")
    import traceback
    traceback.print_exc()