# ============================================================
# PatchDiscriminator.py
# Single-generator CT → PET architecture
# ============================================================

import torch
from torch import nn


class DiscConvBlock(nn.Module):

    def __init__(
        self,
        channels_in,
        channels_out,
        stride=2,
        is_first=False
    ):
        super(DiscConvBlock, self).__init__()

        layers = [
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=4,
                stride=stride,
                padding=1
            ),
            nn.InstanceNorm2d(channels_out),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # The first discriminator block does not use
        # Instance Normalization.
        if is_first:
            layers = [
                layers[0],
                layers[2]
            ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ============================================================
# PatchGAN discriminator for the PET domain
# ============================================================

def PatchGAN(input_channels=3):

    """
    PatchGAN discriminator for single-direction CT → PET
    image translation.

    Input:
        Real PET image y
        or
        Synthetic PET image G(x_CT)

    Output:
        Patch-level real/fake predictions.

    The discriminator operates only in the PET domain.
    """

    model = nn.Sequential(

        # Block 1
        DiscConvBlock(
            input_channels,
            64,
            is_first=True
        ),

        # Block 2
        DiscConvBlock(
            64,
            128
        ),

        # Block 3
        DiscConvBlock(
            128,
            256
        ),

        # Block 4
        DiscConvBlock(
            256,
            512,
            stride=1
        ),

        # Patch-level real/fake prediction
        nn.Conv2d(
            512,
            1,
            kernel_size=4,
            stride=1,
            padding=1
        )
    )

    return model


# ============================================================
# PixelGAN discriminator
# ============================================================

def PixelDiscriminator(input_channels=3):

    """
    1×1 PixelGAN discriminator for the PET domain.

    This is an alternative to PatchGAN and should be used only
    if explicitly selected in the configuration.
    """

    model = nn.Sequential(

        nn.Conv2d(
            input_channels,
            64,
            kernel_size=1,
            stride=1,
            padding=0
        ),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(
            64,
            128,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        ),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(
            128,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
    )

    return model


# ============================================================
# Discriminator model selector
# ============================================================

def get_model(model_name, input_channels=3):

    if model_name == "patchGAN":

        return PatchGAN(
            input_channels=input_channels
        )

    elif model_name == "pixelGAN":

        return PixelDiscriminator(
            input_channels=input_channels
        )

    else:

        raise NotImplementedError(
            f"{model_name} is not implemented. "
            "Please check for spelling."
        )