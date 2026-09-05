import pandas as pd
from utils import plot_graph, get_proper_df
import os
import train_config


if __name__ == '__main__':

    # Get configuration
    config = train_config.config

    # Plot path
    plot_path = os.path.join(config["model_name"], "Plot and CSV")

    # Create output directory
    os.makedirs(plot_path, exist_ok=True)

    # Read logged metrics
    loss_df = pd.read_csv(os.path.join(config["model_name"], "metrics.csv"), index_col=0)

    # Print available metrics
    print("Columns of the logged info are:")
    print(list(loss_df["metric"].unique()))

    # Get configured metrics
    metric_list = config.get('metrics', [])

    # Check metric names
    for metric in metric_list:
        assert metric.lower() in ['ssim', 'fid', 'psnr'], f"{metric} is not supported"

    # Convert metric names to lowercase
    metric_list = [metric.lower() for metric in metric_list]

    # Define loss plots
    metrics = {"gen_loss_epoch": "Total Generator Loss", "gan_loss_epoch": "Adversarial Loss", "recon_loss_epoch": "Reconstruction Loss", "identity_loss_epoch": "Identity Loss", "dis_loss_epoch": "Discriminator Loss"}

    # Add PET image-quality metrics
    for metric in metric_list:
        metrics[f'{metric}_B_epoch'] = f"{metric.upper()} between Predicted and Actual PET"

    # Generate plots
    for metric in metrics.keys():
        metric_df = get_proper_df(loss_df, metric=metric, step="epoch")

        if metric_df is None or len(metric_df) == 0:
            print(f"Warning: {metric} was not found or contains no logged data.")
            continue

        save_path = os.path.join(plot_path, f"{metric}.png")
        ylabel = "Loss" if "loss" in metric.lower() else metric.split('_')[0].upper()
        plot_graph(metric_df["epoch"], metric_df["value"], xlabel="# Epoch", ylabel=ylabel, title=metrics[metric], save_path=save_path)
        print(f"Saved: {save_path}")