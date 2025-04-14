# Reproduction-Study-KPCA
Code for "A Reproduction Study: The Kernel PCA Interpretation of Self-Attention Fails Under Scrutiny". 




### 1) Reconstruction Plots

The code to plot the reconstruction loss is located in the `Reconstruction` folder. You can either use the `run.sh` script to run the code or run the `main_train.py` script directly. The code is adapted from the [KPCA repository](https://github.com/rachtsy/KPCA_code).
Important arguments:
- `--dataset`: The dataset to use. Options are `mnist`, `fashion`, `cifar10`, `cifar100`, and `svhn`. Default is `mnist`.
- `--model`: The vision transformer model to use (for example `deit_tiny_patch16_224`)
- `--data-path`: The path to the dataset (such as `/weka/datasets/ImageNet2012`). There should be subfolders `train` and `val` with the images in them. Alternatively, you can modify the necessary parts in `datasets.py` to make it compatible with your setup. 
- `--output_dir`: The path to the output directory where the model checkpoints and logs will be saved. Default is `checkpoints`.
- `--wandb`: Whether to use Weights and Biases for logging. Default is `False`. If you want to log to Wandb, you need to set the `WANDB_API_KEY` environment variable to your Wandb API key. You can also set the `entity`, `project` and `run_name` arguments to specify the Wandb entity, project and run name.

### 2) Value Vectors Convergence Test
