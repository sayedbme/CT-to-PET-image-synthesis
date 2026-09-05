# ============================================================
# ResnetGenerator.py
# Single-generator CT → PET image translation
# ============================================================
import torch
from torch import nn

# ============================================================
# Residual Block
# ============================================================

class ResnetBlock(nn.Module):
    """
    Residual block consisting of two 3×3 convolutional layers.

    The block learns a residual mapping:
        output = F(x) + x
    """

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        self.conv_block = nn.Sequential(

            # First convolution
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            # Second convolution
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


# ============================================================
# Downsampling Block
# ============================================================

def ConvBlock(channels_out):

    channels_in = channels_out // 2

    return (
        nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=2,
            padding=1
        ),
        nn.InstanceNorm2d(channels_out),
        nn.ReLU(inplace=True)
    )


# ============================================================
# Upsampling Block
# ============================================================

def ConvTransposeBlock(channels_out):

    channels_in = channels_out * 2

    return (
        nn.ConvTranspose2d(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        ),
        nn.InstanceNorm2d(channels_out),
        nn.ReLU(inplace=True)
    )


# ============================================================
# ResNet Generator
# ============================================================

def Resnet_Gen(
    num_residual_blocks=9,
    input_channels=3,
    output_channels=3
):
    """
    Single-direction CT → PET generator.

    Input:
        CT image x

    Output:
        Synthetic PET image G(x)

    Parameters
    ----------
    num_residual_blocks : int
        Number of residual blocks.

    input_channels : int
        Number of CT input channels.

    output_channels : int
        Number of PET output channels.
    """

    model = nn.Sequential(

        # ----------------------------------------------------
        # Initial feature extraction
        # ----------------------------------------------------
        nn.ReflectionPad2d(3),

        nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=1,
            padding=0
        ),

        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),


        # ----------------------------------------------------
        # Downsampling
        # ----------------------------------------------------
        *ConvBlock(128),
        *ConvBlock(256),


        # ----------------------------------------------------
        # Residual blocks
        # ----------------------------------------------------
        *[
            ResnetBlock(256)
            for _ in range(num_residual_blocks)
        ],


        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------
        *ConvTransposeBlock(128),
        *ConvTransposeBlock(64),


        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------
        nn.ReflectionPad2d(3),

        nn.Conv2d(
            64,
            output_channels,
            kernel_size=7,
            stride=1,
            padding=0
        ),

        # Output normalized to [-1, 1]
        nn.Tanh()
    )

    return model


# ============================================================
# Specific ResNet generator configurations
# ============================================================

def Resnet_Gen_9(input_channels=3, output_channels=3):

    return Resnet_Gen(
        num_residual_blocks=9,
        input_channels=input_channels,
        output_channels=output_channels
    )


def Resnet_Gen_7(input_channels=3, output_channels=3):

    return Resnet_Gen(
        num_residual_blocks=7,
        input_channels=input_channels,
        output_channels=output_channels
    )


def Resnet_Gen_3(input_channels=3, output_channels=3):

    return Resnet_Gen(
        num_residual_blocks=3,
        input_channels=input_channels,
        output_channels=output_channels
    )


# ============================================================
# Generator selector
# ============================================================

def get_generator(
    model_name,
    input_channels=3,
    output_channels=3
):

    if model_name == "resnet_gen_9":

        return Resnet_Gen_9(
            input_channels=input_channels,
            output_channels=output_channels
        )

    elif model_name == "resnet_gen_7":

        return Resnet_Gen_7(
            input_channels=input_channels,
            output_channels=output_channels
        )

    elif model_name == "resnet_gen_3":

        return Resnet_Gen_3(
            input_channels=input_channels,
            output_channels=output_channels
        )

    else:

        raise NotImplementedError(
            f"{model_name} is not implemented. "
            "Please check for spelling."
        )