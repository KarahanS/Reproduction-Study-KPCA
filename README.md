# Reproduction-Study-KPCA
Code for "A Reproduction Study: The Kernel PCA Interpretation of Self-Attention Fails Under Scrutiny". 


### Ideal dataset format:
The code is designed to work with the ImageNet dataset. The dataset should be organized in the following way:
```bash
├── ImageNet
│   ├── train
│   │   ├── n01440764
│   │   │   ├── n01440764_10026.JPEG
│   │   │   ├── ... (more images)
│   │   ├── n01443537
│   │   │   ├── n01443537_12345.JPEG
│   │   │   ├── ... (more images)
│   │   └── ... (more class folders)
│   ├── val
│   │   ├── n01440764
│   │   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   │   ├── ILSVRC2012_val_00000294.JPEG
│   │   │   └── ... (more images)
│   │   ├── n01443537
│   │   │   ├── ILSVRC2012_val_00000345.JPEG
│   │   │   └── ... (more images)
│   │   └── ... (more class folders)
```

### 1) Reconstruction Loss (Training)

The code to plot the reconstruction loss is located in the `Reconstruction` folder. You can either use the `run.sh` script to run the code or run the `main_train.py` script directly. The code is adapted from the original [KPCA repository](https://github.com/rachtsy/KPCA_code).
Important arguments:
- `--model`: The vision transformer model to use (for example `deit_tiny_patch16_224`)
- `--data-path`: The path to the dataset (such as `/weka/datasets/ImageNet2012`). There should be subfolders `train` and `val` with the images in them. Alternatively, you can modify the necessary parts in `datasets.py` to make it compatible with your setup. 
- `--output_dir`: The path to the output directory where the model checkpoints and logs will be saved. Default is `checkpoints`.
- `--wandb`: Whether to use Weights and Biases for logging. Default is `False`. If you want to log to Wandb, you need to set the `WANDB_API_KEY` environment variable to your Wandb API key. You can also set the `entity`, `project` and `run_name` arguments to specify the Wandb entity, project and run name.

You can reproduce our results (for DeiT-Tiny) as follows (within the `Reconstruction` folder):
```bash
python main_train.py \
    --model deit_tiny_patch16_224 \
    --batch-size 256 \
    --data-path "PATH_TO_IMAGENET" \
    --output_dir "checkpoints" \
    --lr 1e-4 \
    --warmup-epochs 0 \
    --wandb \
    --wandb_project "PROJECT_NAME" \
    --wandb_entity "ENTITY_NAME" \
    --wandb_run_name "RUN_NAME" \
```

This will log the reconstruction error for a single batch (selected from the training set) to Wandb. Additionally we are saving the squared norms of the query projections and attention outputs individually to have a better overview of the behavior of the model. 


### 2) Eigenvalues Analysis

To reproduce the eigenvalue analysis results, please use the `eigenvalues.py` script. We randomly sample images from the ImageNet validation set and compute the eigenvalues of the gram matrix for each model. Although we fixed the seed for exact reproducibility, results may differ based on your dataset folder structure.

You can change the `standardize` argument to `True` or `False` to see the effect of standardization on the eigenvalues. The default is `True`. If set to `False`, output may be `NaN` for some models.

### 3) Squared Norms of $\varphi(\mathbf{q}_i)$ and $\mathbf{h}_i$

To visualize the squared norms of $\varphi(\mathbf{q}_i)$ and $\mathbf{h}_i$ for each layer (as in Appendix A), you can use the `norms.py` script$. This script generates a layer-wise comparative plot of two norm metrics across the residual blocks (layers), visualizing their statistical distributions.

### 4) $| \gamma_i - \gamma_j|$ Analysis
To reproduce the $|\gamma_i - \gamma_j|$ plot from our paper, run the `gamma.py` script: it generates distributions of both absolute and relative differences across principal component indices.

### 5) Value Vectors Convergence Analysis
To analyze convergence between self-attention learned value vectors and KPCA-derived theoretical values, use the `value_vectors.py` script. Configure different models in `config.yaml` to compare results across architectures. This generates statistical measures of alignment between empirical and theoretical feature representations.

### 6) Reconstruction Plots
You can use `reconstruction.py` to generate the reconstruction plots with train/test error and individual norm values. Results of our experiments are stored as `csv` files in the `wandb_data` folder. The script will automatically load the data from the `wandb_data` folder and plot the results.

You can use `reconstruction_relative.py` to visualize the relative average absolute error (as in Appendix B). This script will also load the data from the `wandb_data` folder and plot the results, similar to the script above.

### 7) Language Models

You can use `nlp_norms.py` to analyze the squared norms of the query projections and attention outputs in encoder-only language models, similar to the functionality of `norms.py`. To compute similarity metrics between the self-attention value vectors learned by the model and those derived theoretically using KPCA, use `nlp_value_vectors.py`. Our experiments were conducted on the `WikiText-103` dataset, which will be automatically downloaded by `nlp_value_vectors.py`.
