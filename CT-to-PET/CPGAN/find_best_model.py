# ============================================================
# Imports
# ============================================================
import pandas as pd
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm

import infer_config_updated
from PairedDataset import PairedDataset
from cyclegan_models import CycleGan
from utils import convert_to_255
from metrics import get_metric_class, calc_metric

import warnings

warnings.filterwarnings(
    "ignore",
    ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------------
    config = infer_config_updated.config
    root = config["data_path"]
    mode = config["sub_fold"]

    # --------------------------------------------------------
    # 2. Metrics
    # --------------------------------------------------------
    if not config["paired"]:
        raise NotImplementedError(
            "Metrics for Unpaired Data are not implemented."
        )

    metric_list = config["metrics"]

    # Check whether requested metrics are supported
    for metric in metric_list:
        assert metric.lower() in [
            "ssim",
            "fid",
            "psnr",
        ], f"{metric} is not supported"

    # Convert metric names to lowercase
    metric_list = [metric.lower() for metric in metric_list]

    # --------------------------------------------------------
    # 3. Create paired test dataset
    # --------------------------------------------------------
    test_ds = PairedDataset(root, mode)

    # --------------------------------------------------------
    # 4. Create DataLoader
    # --------------------------------------------------------
    test_dl = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False
    )

    # --------------------------------------------------------
    # 5. Select device
    # --------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}")

    # --------------------------------------------------------
    # 6. Find all checkpoints
    # --------------------------------------------------------
    model_list = sorted([
        f
        for f in os.listdir(config["model_name"])
        if f.endswith(".ckpt")
    ])

    print(f"\nTotal {len(model_list)} models found\n")


    all_results = []

    # ========================================================
    # 8. Evaluate every checkpoint
    # ========================================================
    for model_name in model_list:

        model_path = os.path.join(
            config["model_name"],
            model_name
        )

        print(
            f"Loading model from -> {model_path}\n"
        )

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------
        model = CycleGan.load_from_checkpoint(
            model_path,
            map_location=device
        )

        model.eval()
        model.to(device)

        # ----------------------------------------------------
        # Create metric objects
        # ----------------------------------------------------
        metric_dict = {}

        for metric in metric_list:
            metric_dict[metric] = get_metric_class(
                metric
            ).to(device)

        # ====================================================
        # 9. Iterate through test batches
        # ====================================================
        for batch_idx, batch in enumerate(
            tqdm(
                test_dl,
                total=len(test_dl),
                desc=f"Evaluating {model_name}"
            )
        ):

            # ------------------------------------------------
            # CT input and ground-truth PET
            # ------------------------------------------------
            imgA = batch["A"].to(device)   # CT
            imgB = batch["B"].to(device)   # Ground-truth PET

            # ------------------------------------------------
            # Single-generator inference
            #
            # CT --> Generator --> Synthetic PET
            # ------------------------------------------------
            with torch.no_grad():
                fakeB = model.generator(imgA)

            # ------------------------------------------------
            # Convert images to [0, 255]
            # ------------------------------------------------
            imgB_255 = convert_to_255(imgB)
            fakeB_255 = convert_to_255(fakeB)

            # ------------------------------------------------
            # Calculate CT --> PET metrics only
            # ------------------------------------------------
            for metric in metric_list:

                metric_dict[metric] = calc_metric(
                    metric_dict[metric],
                    metric,
                    fakeB_255,
                    imgB_255
                )

        # ====================================================
        # 10. Print + collect every metric for this model
        # ====================================================
        print(f"\n{'#' * 60}")
        print(f"Model: {model_name}")
        print(f"{'#' * 60}")

        model_metric_values = {"Model": model_name}

        for metric in metric_list:

            metric_value = metric_dict[metric].compute()

            model_metric_values[metric.upper()] = metric_value.item()

            print(
                f"{metric.upper()} (CT -> PET) = "
                f"{metric_value:.3f}"
            )

        all_results.append(model_metric_values)

    # ========================================================
    # 11. Save results (all configured metrics, every model)
    # ========================================================
    df = pd.DataFrame(all_results)

    csv_path = os.path.join(
        config["model_name"],
        "CT_to_PET_metrics.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # ========================================================
    # 12. Identify best model (ranked by the first configured metric)
    # ========================================================
    primary_metric = metric_list[0].upper()

    # For SSIM and PSNR, higher is better.
    if metric_list[0] in ["ssim", "psnr"]:

        best_idx = df[primary_metric].idxmax()

    # For FID, lower is better.
    elif metric_list[0] == "fid":

        best_idx = df[primary_metric].idxmin()

    else:
        best_idx = 0

    print("\n" + "=" * 60)
    print("CT -> PET EVALUATION COMPLETED")
    print("=" * 60)

    print(
        f"Best model: {df.loc[best_idx, 'Model']}"
    )

    print(
        f"Best {primary_metric}: "
        f"{df.loc[best_idx, primary_metric]:.3f}"
    )

    print(
        f"\nResults saved to:\n{csv_path}"
    )

    print("\nInference Completed!")
