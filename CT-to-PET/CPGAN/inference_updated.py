# Imports
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm

# Pipeline Imports
import infer_config_updated
from PairedDataset import PairedDataset
from cyclegan_models import CycleGan
from utils import save_paired_image


import warnings
warnings.filterwarnings(
    "ignore",
    ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    config = infer_config_updated.config
    root = config["data_path"]
    mode = config["sub_fold"]

    # ---------------------------------------------------------
    # 2. Create dataset (paired CT -> PET only)
    # ---------------------------------------------------------
    if not config["paired"]:
        raise NotImplementedError(
            "This single-generator CT-to-PET implementation supports paired data only."
        )

    test_ds = PairedDataset(root, mode)

    # ---------------------------------------------------------
    # 3. Create DataLoader
    # ---------------------------------------------------------
    test_dl = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False
    )

    # ---------------------------------------------------------
    # 4. Select device
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 5. Iterate through checkpoints
    # ---------------------------------------------------------
    for ckpt_name in config["ckpt_names"]:

        ckpt_path = os.path.join(
            config["model_name"],
            ckpt_name
        )

        print(f"\nLoading model from -> {ckpt_path}")

        # Load trained model
        model = CycleGan.load_from_checkpoint(ckpt_path, map_location=device)

        # Evaluation mode
        model.eval()
        model.to(device)

        # -----------------------------------------------------
        # Extract epoch number
        # -----------------------------------------------------
        try:
            epoch_number = ckpt_name.split("=")[1].split("-")[0]
        except (IndexError, ValueError):
            epoch_number = os.path.splitext(ckpt_name)[0]

        # -----------------------------------------------------
        # Output directory
        # -----------------------------------------------------
        path_dir = os.path.join(
            config["model_name"],
            f"Epoch_{epoch_number}_{config['sub_fold']}_CT2PET_Predictions"
        )

        os.makedirs(path_dir, exist_ok=True)

        # -----------------------------------------------------
        # 6. Inference
        # -----------------------------------------------------
        with torch.no_grad():

            for batch_idx, batch in enumerate(
                tqdm(test_dl, total=len(test_dl))
            ):


                imgA = batch["A"].to(device)
                imgB = batch["B"].to(device)

                # -------------------------------------------------
                # Single generator:
                # CT -> PET
                # -------------------------------------------------
                fakeB = model.generator(imgA)

                # -------------------------------------------------
                # Move tensors to CPU
                # -------------------------------------------------
                imgA = imgA.cpu()
                imgB = imgB.cpu()
                fakeB = fakeB.cpu()

                # -------------------------------------------------
                # Get CT image paths
                # -------------------------------------------------
                pathA = batch["pathA"]

                # -------------------------------------------------
                # Save CT / ground-truth PET / synthetic PET
                # -------------------------------------------------
                for sample_idx in range(len(pathA)):

                    temp_pathA = os.path.basename(pathA[sample_idx])

                    save_path = os.path.join(
                        path_dir,
                        temp_pathA
                    )

                    save_paired_image(
                        [imgA[sample_idx]],
                        [imgB[sample_idx]],
                        [fakeB[sample_idx]],
                        save_path
                    )

        print(
            f"Inference completed for epoch {epoch_number}!"
        )
