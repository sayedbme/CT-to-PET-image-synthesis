# utils.py

import torch
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def init_weights(net, init_type="normal", init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (nn.Module): network to be initialized
        init_type (str): normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for initialization
    """

    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1
            or classname.find("Linear") != -1
        ):

            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == "kaiming":
                init.kaiming_normal_(
                    m.weight.data,
                    a=0,
                    mode="fan_in"
                )

            elif init_type == "orthogonal":
                init.orthogonal_(
                    m.weight.data,
                    gain=init_gain
                )

            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented"
                    % init_type
                )

            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm2d") != -1:

            init.normal_(
                m.weight.data,
                1.0,
                init_gain
            )

            init.constant_(
                m.bias.data,
                0.0
            )

    net.apply(init_func)


def set_requires_grad(nets, requires_grad):
    """
    Enable or disable gradient computation for networks.
    """

    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def load_weights(ckpt, net):
    """
    Load model weights from a PyTorch Lightning checkpoint.
    """

    target_state = torch.load(
        ckpt,
        map_location="cpu"
    )

    target_state = target_state["state_dict"]

    net.load_state_dict(target_state)

    return net


def plot_graph(
    x,
    y,
    xlabel,
    ylabel,
    title,
    save_path
):
    """
    Plot and save a training/evaluation curve.
    """

    plt.figure(figsize=(10, 10))

    plt.plot(x, y)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    plt.savefig(
        save_path,
        dpi=100,
        bbox_inches="tight"
    )

    plt.close()


def toImage(x):
    """
    Convert a tensor in [-1, 1] to a PIL RGB image.
    """

    x = x.detach().cpu().numpy()

    x = np.transpose(
        x,
        (1, 2, 0)
    )

    x = (x + 1.0) * 127.5

    x = np.clip(
        x,
        0,
        255
    ).astype("uint8")

    return Image.fromarray(x)


def convert_to_255(x: torch.Tensor):
    """
    Convert a tensor from [-1, 1] to [0, 255].
    """

    x = (x + 1.0) * 127.5

    x = torch.clamp(
        x,
        0,
        255
    )

    return x.to(torch.uint8)


def save_paired_image(
    imgCT,
    imgPET,
    fakePET,
    path
):
    """
    Save paired CT -> PET translation results.

    Layout:
        [CT | Synthetic PET]
        [Ground-truth PET | Synthetic PET]

    This function contains no PET -> CT translation.
    """

    imgCT = [
        toImage(f)
        for f in imgCT
    ]

    imgPET = [
        toImage(f)
        for f in imgPET
    ]

    fakePET = [
        toImage(f)
        for f in fakePET
    ]

    width = imgCT[0].size[0]
    height = imgCT[0].size[1]

    new_image = Image.new(
        "RGB",
        (2 * width, 2 * height),
        (250, 250, 250)
    )

    # CT
    new_image.paste(
        imgCT[0],
        (0, 0)
    )

    # Synthetic PET
    new_image.paste(
        fakePET[0],
        (width, 0)
    )

    # Ground-truth PET
    new_image.paste(
        imgPET[0],
        (0, height)
    )

    # Synthetic PET
    new_image.paste(
        fakePET[0],
        (width, height)
    )

    new_image.save(path)


def tflog2pandas(path):
    """
    Convert TensorBoard scalar logs into a pandas DataFrame.
    """

    runlog_data = pd.DataFrame(
        {
            "metric": [],
            "value": [],
            "step": []
        }
    )

    try:

        event_acc = EventAccumulator(path)

        event_acc.Reload()

        tags = event_acc.Tags()["scalars"]

        for tag in tags:

            event_list = event_acc.Scalars(tag)

            values = list(
                map(
                    lambda x: x.value,
                    event_list
                )
            )

            step = list(
                map(
                    lambda x: x.step,
                    event_list
                )
            )

            r = {
                "metric": [tag] * len(step),
                "value": values,
                "step": step
            }

            r = pd.DataFrame(r)

            runlog_data = pd.concat(
                [
                    runlog_data,
                    r
                ],
                ignore_index=True
            )

    except Exception:

        print(
            "Event file possibly corrupt: {}".format(path)
        )

        traceback.print_exc()

    return runlog_data


def get_proper_df(
    df,
    metric,
    step="epoch"
):
    """
    Extract a particular metric from TensorBoard logs.
    """

    columns = list(
        df["metric"].unique()
    )

    assert metric not in [
        "epoch",
        "step"
    ], f"{metric} cannot be 'epoch' or 'step'"

    assert metric in columns, (
        f"{metric} is not a valid metric"
    )

    assert step in [
        "epoch",
        "step"
    ], (
        f"{step} should be either 'epoch' or 'step'"
    )

    assert metric.split("_")[-1] == step, (
        f"{metric} is incompatible with step = {step}"
    )

    idx = df["metric"] == metric

    temp = df[idx].copy()

    if step == "epoch":

        idx = df["metric"] == step

        epoch = df[idx].copy()

        epoch.columns = [
            "metric",
            "epoch",
            "step"
        ]

        temp = (
            pd.merge(
                temp,
                epoch[["epoch", "step"]],
                on="step",
                how="left"
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

        temp["epoch"] = temp.index

    temp = temp.drop(
        "metric",
        axis=1
    )

    return temp