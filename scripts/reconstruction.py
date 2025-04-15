import numpy as np
import matplotlib.pyplot as plt
import os
from tueplots import bundles
import pandas as pd

# Set up plot styling with tueplots
plt.rcParams.update(bundles.icml2024(column="half", nrows=1, ncols=1))
plt.rcParams.update({"xtick.labelsize": 40})
plt.rcParams.update({"axes.labelsize": 40})
plt.rcParams.update({"ytick.labelsize": 40})
plt.rcParams.update({"axes.titlesize": 45})
plt.rcParams.update({"legend.fontsize": 40})
plt.rcParams.update({"font.size": 35})
plt.rcParams.update({"legend.title_fontsize": 45})
plt.rcParams.update({"axes.titlepad": 25})  # Set global title padding
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['cmss10']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{cmbright}'
# Configuration
# Epoch to start plotting from (to avoid outliers)
start_epoch = 10  
start_step = 100  # 100 steps explicitly

# Base paths
data_root = "wandb_data"  # Base folder
model_folders = ["deit_tiny", "vit_tiny"]

# File paths
file_paths = {
    "DeiT-Tiny": {
        "train_error": f"{data_root}/deit_tiny/deit_train_error.csv",
        "test_error": f"{data_root}/deit_tiny/deit_test_error.csv",
        "train_h_norms": f"{data_root}/deit_tiny/deit_train_h.csv",
        "test_h_norms": f"{data_root}/deit_tiny/deit_test_h.csv",
        "train_phiq_norms": f"{data_root}/deit_tiny/deit_train_varphi.csv",
        "test_phiq_norms": f"{data_root}/deit_tiny/deit_test_varphi.csv"
    },
    "ViT-Tiny": {
        "train_error": f"{data_root}/vit_tiny/vit_train_error.csv",
        "test_error": f"{data_root}/vit_tiny/vit_test_error.csv",
        "train_h_norms": f"{data_root}/vit_tiny/vit_train_h.csv",
        "test_h_norms": f"{data_root}/vit_tiny/vit_test_h.csv",
        "train_phiq_norms": f"{data_root}/vit_tiny/vit_train_varphi.csv",
        "test_phiq_norms": f"{data_root}/vit_tiny/vit_test_varphi.csv"
    }
}

# Model configurations with colors
model_info = [
    {
        "model_name": "DeiT-Tiny", 
        "color_base": {
            "train": "#FF9999",  # Light red for train
            "test": "#CC0000",   # Dark red for test
            "train_h": "#FF9999",      # Light red for train_h
            "test_h": "#CC0000",       # Dark red for test_h
            "train_phiq": "#FF6666",   # Medium red for train_phiq
            "test_phiq": "#990000"     # Very dark red for test_phiq
        },
        "markers": {
            "train_h": "o",            # Circle for train_h
            "test_h": "o",             # Circle for test_h
            "train_phiq": "^",         # Triangle up for train_phiq
            "test_phiq": "^"           # Triangle up for test_phiq
        },
        "markersize": 6,  # Normal markers
        "triangle_markersize": 15  # Larger triangles
    },
    {
        "model_name": "ViT-Tiny", 
        "color_base": {
            "train": "#99CCFF",  # Light blue for train
            "test": "#0066CC",   # Dark blue for test
            "train_h": "#99CCFF",      # Light blue for train_h
            "test_h": "#0066CC",       # Dark blue for test_h
            "train_phiq": "#66AAFF",   # Medium blue for train_phiq
            "test_phiq": "#003399"     # Very dark blue for test_phiq
        },
        "markers": {
            "train_h": "o",            # Circle for train_h
            "test_h": "o",             # Circle for test_h
            "train_phiq": "^",         # Triangle up for train_phiq
            "test_phiq": "^"           # Triangle up for test_phiq
        },
        "markersize": 6,  # Normal markers
        "triangle_markersize": 15  # Larger triangles
    }
]

# Function to read CSV files
def read_csv_file(file_path):
    try:
        # First, try to read assuming standard CSV format
        df = pd.read_csv(file_path)
        
        # If the CSV has no header, try to infer column names
        if len(df.columns) == 1:  # Single column detected
            df = pd.read_csv(file_path, header=None, names=["combined"])
            # Try to split based on whitespace or delimiter
            if df["combined"].dtype == object:  # It's a string
                # Split by various delimiters
                parts = df["combined"].str.split(r'\s+|\t|,', expand=True)
                if len(parts.columns) >= 2:
                    df = parts
                    df.columns = ["Step", "Value"]
        
        # Ensure we have expected columns
        if "Step" not in df.columns and len(df.columns) >= 2:
            # Assume first column is Step, second is Value
            df.columns = ["Step"] + [f"Value{i}" for i in range(1, len(df.columns))]
            
        # Convert Step to numeric if possible
        if "Step" in df.columns:
            df["Step"] = pd.to_numeric(df["Step"], errors='coerce')
            
        # Extract the value column
        value_col = [col for col in df.columns if col != "Step"][0] if "Step" in df.columns else df.columns[1]
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Rename columns if needed
        if "Step" in df.columns and value_col != "Value":
            df = df.rename(columns={value_col: "Value"})
        elif "Step" not in df.columns and len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: "Step", df.columns[1]: "Value"})
        
        # Add epoch column (15 steps = 1 epoch)
        df['epoch'] = df['Step'] / 15
        
        return df
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=["Step", "Value", "epoch"])

# Create the figure
plt.figure(figsize=(15, 9))

# Track overall min/max values for axis scaling
overall_y_min, overall_y_max = float('inf'), float('-inf')

# Read data and plot for each model
for model in model_info:
    model_name = model["model_name"]
    colors = model["color_base"]
    markers = model["markers"]
    markersize = model["markersize"]
    triangle_markersize = model["triangle_markersize"]
    
    print(f"\n===== Processing {model_name} data =====")
    
    # Process error metrics first (line plots)
    for metric_type in ["train_error", "test_error"]:
        file_path = file_paths[model_name][metric_type]
        print(f"Reading {metric_type} from {file_path}")
        
        # Read the file
        df = read_csv_file(file_path)
        
        if not df.empty:
            # Filter data to start from the specified epoch
            df = df[df['epoch'] >= start_epoch]
            
            # Skip if no data after filtering
            if df.empty:
                print(f"No data for {metric_type} after filtering to epoch >= {start_epoch}")
                continue
                
            # Sort by epoch to ensure correct line connections
            df = df.sort_values(by='epoch')
            
            # Define line style
            line_style = {
                "color": colors["train"] if "train" in metric_type else colors["test"],
                "linestyle": "-",
                "linewidth": 2,
                "label": f"{model_name} {'Train' if 'train' in metric_type else 'Test'} Error"
            }
            
            # Plot the line
            plt.plot(df['epoch'], df['Value'], **line_style)
            
            # Update min/max for axis scaling
            overall_y_min = min(overall_y_min, df['Value'].min())
            overall_y_max = max(overall_y_max, df['Value'].max())
            
            print(f"Plotted {metric_type}: {len(df)} points (range: {df['Value'].min():.4f} to {df['Value'].max():.4f})")
        else:
            print(f"No valid data for {metric_type}")
    
    # Process norm metrics (marker plots)
    for metric_type in ["train_h_norms", "test_h_norms", "train_phiq_norms", "test_phiq_norms"]:
        file_path = file_paths[model_name][metric_type]
        print(f"Reading {metric_type} from {file_path}")
        
        # Read the file
        df = read_csv_file(file_path)
        
        if not df.empty:
            # Filter data to start from the specified epoch
            df = df[df['epoch'] >= start_epoch]
            
            # Skip if no data after filtering
            if df.empty:
                print(f"No data for {metric_type} after filtering to epoch >= {start_epoch}")
                continue
                
            # Sort by epoch
            df = df.sort_values(by='epoch')
            
            # Sample every 3rd point to avoid overcrowding
            sample_step = 3
            sampled_df = df.iloc[::sample_step]
            
            # Get the right marker size for this metric
            current_markersize = triangle_markersize if "phiq" in metric_type else markersize
            
            # Get color key for this metric
            color_key = metric_type.replace("_norms", "")
            
            # Define marker style
            marker_style = {
                "color": colors[color_key],
                "marker": markers["train_phiq" if "phiq" in metric_type else "train_h"],  # Same marker for train/test
                "linestyle": "none",
                "markersize": current_markersize,
                "label": f"{model_name} {'Train' if 'train' in metric_type else 'Test'} {'PhiQ' if 'phiq' in metric_type else 'H'} Norm",
                "alpha": 0.7
            }
            
            # Plot the markers
            plt.plot(sampled_df['epoch'], sampled_df['Value'], **marker_style)
            
            # Update min/max for axis scaling
            overall_y_min = min(overall_y_min, df['Value'].min())
            overall_y_max = max(overall_y_max, df['Value'].max())
            
            print(f"Plotted {metric_type}: {len(sampled_df)} points (range: {df['Value'].min():.4f} to {df['Value'].max():.4f})")
        else:
            print(f"No valid data for {metric_type}")
            
# Adjust y-axis range with some padding
if overall_y_min != float('inf') and overall_y_max != float('-inf'):
    padding = (overall_y_max - overall_y_min) * 0.1
    plt.ylim(-0.5, overall_y_max + padding)  # Start y-axis at 0
    print(f"\nSetting overall y-axis limits to 0 - {overall_y_max + padding:.4f}")

# Find the maximum epoch
max_epoch = 0
for model in model_info:
    model_name = model["model_name"]
    for metric_type in ["train_error", "test_error", "train_h_norms", "test_h_norms", "train_phiq_norms", "test_phiq_norms"]:
        file_path = file_paths[model_name][metric_type]
        try:
            df = read_csv_file(file_path)
            if not df.empty:
                max_epoch = max(max_epoch, df['epoch'].max())
        except:
            pass

# Create x-axis ticks every 10 epochs, starting from start_epoch
x_ticks = np.arange(start_epoch, max_epoch + 10, 10)
plt.xticks(x_ticks)

# Create y-axis ticks every 1 unit
y_range = int(np.ceil(overall_y_max)) + 1
y_ticks = np.arange(0, y_range, 1)
plt.yticks(y_ticks)

# Set plot labels and title with LaTeX formatting
plt.xlabel('Epochs')
plt.title("Reconstruction Loss")
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Set y-label with LaTeX formatting
plt.ylabel(r'$J_{\text{proj}}$')

# Add grid
plt.grid(True, alpha=0.3)

# Create custom legend
from matplotlib.lines import Line2D
legend_markersize = 12
legend_linewidth = 2
# Create the custom legend items
custom_lines = [
    # Model names row
    Line2D([0], [0], marker='>', color='black', markersize=8, linestyle='none'),  # ViT-Tiny
    Line2D([0], [0], marker='>', color='black', markersize=8, linestyle='none'),  # DeiT-Tiny
    Line2D([0], [0], color='black', lw=0),  # Placeholder for squared norms text
    
    # Train row
    Line2D([0], [0], color=model_info[1]['color_base']['train'], lw=2),           # ViT Train Error
    Line2D([0], [0], color=model_info[0]['color_base']['train'], lw=2),           # DeiT Train Error
    Line2D([0], [0], marker=model_info[0]['markers']['train_h'], color='black', 
           markersize=10, linestyle='none'),           # Circle for H norm
    
    # Test row
    Line2D([0], [0], color=model_info[1]['color_base']['test'], lw=2),            # ViT Test Error
    Line2D([0], [0], color=model_info[0]['color_base']['test'], lw=2),            # DeiT Test Error
    Line2D([0], [0], marker=model_info[0]['markers']['train_phiq'], color='black', 
           markersize=10, linestyle='none'),    # Triangle for PhiQ norm
]

custom_labels = [
    'ViT-Tiny:', 'DeiT-Tiny:', 'Squared norms:',
    'Train', 'Train', r'$\left\|\mathbf{h}_i\right\|^2$',
    'Test', 'Test', r'$\left\|\varphi(\mathbf{q}_i)\right\|^2$'
]

# Display the legend with the organized layout
legend = plt.legend(custom_lines, custom_labels, 
                   loc='upper right',
                   ncol=3,  # Using 3 columns
                   fontsize=24,
                   framealpha=0.9,
                   columnspacing=1.0,
                   handlelength=1.5,
                   handletextpad=0.5,
                   borderpad=0.8)

# Set specific colors for the legend labels
legend_texts = legend.get_texts()
for i, text in enumerate(legend_texts):
    if i == 0:  # ViT-Tiny label
        text.set_color('black')
        text.set_weight('bold')
    elif i == 1:  # DeiT-Tiny label
        text.set_color('black')
        text.set_weight('bold')
    elif i == 2:  # Norm labels
        text.set_color('black')

# Save the plot
output_dir = "reconstruction_outputs"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "reconstruction_loss.pdf")
plt.savefig(plot_path, dpi=300)
print(f"\nSaved plot to {plot_path}")

plt.show()