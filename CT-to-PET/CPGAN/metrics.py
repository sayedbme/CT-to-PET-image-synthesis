from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics import PeakSignalNoiseRatio as PSNR
import torch


def get_metric_class(metric_name: str):
    if metric_name == 'fid':
        return FID(feature=768, normalize=False)
    elif metric_name == 'psnr':
        return PSNR()
    elif metric_name == 'ssim':
        return SSIM(data_range=255)
    else:
        raise NotImplementedError(f"{metric_name} is not implemented")


def calc_metric(metric, metric_name: str, fake_image: torch.Tensor, real_image: torch.Tensor):
    if metric_name == 'fid':
        metric.update(real_image, real=True)
        metric.update(fake_image, real=False)
        return metric
    elif metric_name == 'ssim':
        metric.update(fake_image, real_image)
        return metric
    elif metric_name == 'psnr':
        metric.update(fake_image, real_image)
        return metric
    else:
        raise NotImplementedError(f"{metric_name} is not implemented")
