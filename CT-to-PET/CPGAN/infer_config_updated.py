# ============================================================
# Inference Configuration
# Single-generator CT -> PET
# ============================================================

##### DO NOT EDIT THESE LINES #####
config = {}
###################################

#### START EDITING FROM HERE ######

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
config["data_path"] = r"....\Data"

# Keep the images in either "Test" or "Val" subfolder
config["sub_fold"] = "Test"


# ------------------------------------------------------------
# Model / Results directory
# ------------------------------------------------------------
config["model_name"] = (
    r"....\Results\resnet_gen_9_patchGAN_paired"
)


# ------------------------------------------------------------
# Checkpoint(s) to evaluate
# ------------------------------------------------------------

config["ckpt_names"] = [
    "ct2pet-epoch=00194-step=390780.ckpt",
]


# ------------------------------------------------------------
# Dataset type
# ------------------------------------------------------------
# True  = paired CT-PET data
# False = unpaired data (NOT supported by this pipeline)
#
# For your CT -> PET reconstruction loss and quantitative
# evaluation, use paired data.
# ------------------------------------------------------------
config["paired"] = True


# ------------------------------------------------------------
# Batch size
# ------------------------------------------------------------
config["batch_size"] = 2


# ------------------------------------------------------------
# Generator
# ------------------------------------------------------------
# Single generator:
#
#       CT -> G -> PET
#
# Available ResNet variants:
#   resnet_gen_9
#   resnet_gen_7
#   resnet_gen_3
# ------------------------------------------------------------
config["gen_model"] = "resnet_gen_9"


# ------------------------------------------------------------
# Discriminator
# ------------------------------------------------------------
# The discriminator operates only in the PET domain:
#
#       Real PET        -> D_PET
#       Synthetic PET   -> D_PET
#
# Available:
#   patchGAN
#   pixelGAN
# ------------------------------------------------------------
config["dis_model"] = "patchGAN"


# ------------------------------------------------------------
# Metrics to compute
# ------------------------------------------------------------
# SSIM is the primary metric used for selecting the loss
# weighting combination.
#
# Additional metrics:
#   FID
#   PSNR
# ------------------------------------------------------------
config["metrics"] = [
    "ssim",
    "psnr",
    "fid"
]
