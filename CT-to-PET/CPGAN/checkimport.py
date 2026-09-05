from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import os
import shutil
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union
from torch import Tensor
from lightning_fabric.loggers.csv_logs import _ExperimentWriter as _FabricExperimentWriter
from lightning_fabric.loggers.csv_logs import CSVLogger as FabricCSVLogger
from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning_fabric.utilities.logger import _convert_params
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import itertools
import pytorch_lightning as pl