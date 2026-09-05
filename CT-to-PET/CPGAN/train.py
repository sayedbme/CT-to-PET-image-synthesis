# train.py

from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import shutil

import train_config
from PairedDataset import PairedDataset
from cyclegan_models import CycleGan
from utils import plot_graph, tflog2pandas, get_proper_df

import warnings

warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")


if __name__ == "__main__":

    # Get configuration
    config = train_config.config

    # Total epochs
    config["n_epochs"] = config['n_lin_epoch'] + config['n_dec_epoch']

    # Dataset path
    root = config["data_path"]

    # Check paired dataset
    if config['paired']:
        metric_list = config.get('metrics', [])

        for metric in metric_list:
            assert metric.lower() in ['ssim', 'fid', 'psnr'], f"{metric} is not supported"

        metric_list = [metric.lower() for metric in metric_list]
        train_ds = PairedDataset(root, "Train")

    else:
        raise NotImplementedError("This single-generator CT-to-PET implementation supports paired data only.")

    # DataLoader
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False)

    # Resume checkpoint
    if config["resume_ckpt"] is not None:
        resume_ckpt = os.path.join(config["model_name"], config["resume_ckpt"])
        assert os.path.exists(resume_ckpt), f"{resume_ckpt} file does not exist"
    else:
        resume_ckpt = None

    # TensorBoard logger
    logger = pl.loggers.TensorBoardLogger(save_dir=config["model_name"], name="", version="", default_hp_metric=False)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=config["model_name"], filename="ct2pet-{epoch:05d}-{step}", every_n_epochs=config["save_freq"], save_last=True, verbose=True, save_top_k=-1)

    # Instantiate model
    if resume_ckpt is None:
        model = CycleGan(config)
    else:
        model = CycleGan.load_from_checkpoint(resume_ckpt, config=config)

    # Trainer
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=config["n_epochs"], log_every_n_steps=1, default_root_dir=config["model_name"], logger=logger, callbacks=[checkpoint_callback])

    # Train
    trainer.fit(model, train_dl, ckpt_path=resume_ckpt)

    # Copy configuration file
    shutil.copy(src="train_config.py", dst=os.path.join(config["model_name"], "train_config.py"))

    # Paths
    loss_path = config["model_name"]
    plot_path = os.path.join(config["model_name"], "Plot and CSV")

    # Create plot directory
    os.makedirs(plot_path, exist_ok=True)

    # Convert TensorBoard logs to Pandas DataFrame
    loss_df = tflog2pandas(loss_path)

    # Save all logged values
    loss_df.to_csv(os.path.join(config["model_name"], "metrics.csv"))

    print(f"Loss and Metric Plots are saved in -> {plot_path}")

    # Loss plots
    metrics = {"gen_loss_epoch": "Total Generator Loss", "gan_loss_epoch": "Adversarial Loss", "recon_loss_epoch": "Reconstruction Loss", "identity_loss_epoch": "Identity Loss", "dis_loss_epoch": "Discriminator Loss"}

    # Image-quality metrics
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