import torch
import gc
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import pytorch_lightning as pl
import numpy as np
from PIL import Image
import os

from generator import get_generator
from discriminator import get_model
from utils import init_weights, set_requires_grad, convert_to_255
from metrics import get_metric_class, calc_metric


class CycleGan(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.automatic_optimization = False

        # Metrics
        metric_list = config.get('metrics', [])
        self.metric_list = [metric.lower() for metric in metric_list]
        metric_dict = {}
        for metric in self.metric_list:
            metric_dict[metric] = get_metric_class(metric)

        self.metric_dict = torch.nn.ModuleDict(metric_dict)

        # Single CT -> PET generator
        self.generator = get_generator(config['gen_model'])

        # Single PET discriminator
        self.discriminator = get_model(config['dis_model'])

        # Loss weights
        self.lambda_gan = float(config.get('lambda_gan', 1.0))
        self.lambda_recon = float(config.get('lambda_recon', 100.0))
        self.lambda_identity = float(config.get('lambda_identity', 10.0))

        # Logged losses
        self.genLoss = None
        self.disLoss = None
        self.ganLoss = None
        self.reconLoss = None
        self.identityLoss = None
        self.fakeB = None

        # Initialize model weights
        for m in [self.generator, self.discriminator]:
            init_weights(m)

    def configure_optimizers(self):
        optG = Adam(self.generator.parameters(), lr=self.config['gen_lr'], betas=(0.5, 0.999))
        optD = Adam(self.discriminator.parameters(), lr=self.config['dis_lr'], betas=(0.5, 0.999))
        gamma = lambda epoch: 1 - max(0, epoch + 1 - self.config['n_lin_epoch']) / (self.config['n_dec_epoch'] + 1)
        schG = LambdaLR(optG, lr_lambda=gamma)
        schD = LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def get_mse_loss(self, predictions, label):
        """Least-squares GAN loss: real = 1 and fake = 0."""
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        elif label.lower() == 'fake':
            target = torch.zeros_like(predictions)
        else:
            raise ValueError("label must be either 'real' or 'fake'")
        return F.mse_loss(predictions, target)

    def get_reconstruction_loss(self, prediction, target):
        """Compute reconstruction loss between synthesized PET and ground-truth PET."""
        loss_type = self.config.get('recon_loss', 'mae_loss').lower()
        if loss_type == 'mae_loss':
            return F.l1_loss(prediction, target)
        elif loss_type == 'mse_loss':
            return F.mse_loss(prediction, target)
        else:
            raise NotImplementedError(f"Reconstruction loss '{loss_type}' is not implemented.")

    def get_identity_loss(self, prediction, target):
        """Compute identity loss."""
        loss_type = self.config.get('identity_loss', 'mae_loss').lower()
        if loss_type == 'mae_loss':
            return F.l1_loss(prediction, target)
        elif loss_type == 'mse_loss':
            return F.mse_loss(prediction, target)
        else:
            raise NotImplementedError(f"Identity loss '{loss_type}' is not implemented.")

    def generator_training_step(self, imgA, imgB):
        """
        imgA = input CT image x
        imgB = ground-truth PET image y
        fakeB = synthesized PET G(x)
        L_total = lambda_GAN * L_GAN + lambda_recon * L_recon + lambda_identity * L_identity
        """

        x = imgA
        y = imgB

        # CT -> synthesized PET
        fakeB = self.generator(x)

        # L_GAN = MSE(D(G(x)), 1)
        predFakeB = self.discriminator(fakeB)
        ganLoss = self.get_mse_loss(predFakeB, 'real')

        # L_recon = ||G(x) - y||_1
        reconLoss = self.get_reconstruction_loss(fakeB, y)

        # L_identity = ||G(y) - y||_1
        sameB = self.generator(y)
        identityLoss = self.get_identity_loss(sameB, y)

        # L_total = lambda_GAN*L_GAN + lambda_recon*L_recon + lambda_identity*L_identity
        self.genLoss = self.lambda_gan * ganLoss + self.lambda_recon * reconLoss + self.lambda_identity * identityLoss

        # Store individual losses
        self.ganLoss = ganLoss
        self.reconLoss = reconLoss
        self.identityLoss = identityLoss

        # Progress-bar logging
        self.log('_gen_loss', self.genLoss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('_gan_loss', ganLoss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.log('_recon_loss', reconLoss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.log('_identity_loss', identityLoss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=False)

        # TensorBoard logging
        self.log('gen_loss', self.genLoss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('gan_loss', ganLoss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('recon_loss', reconLoss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('identity_loss', identityLoss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # Image-quality metrics
        if self.metric_list:
            self.compute_metrics(imgB, fakeB)

        # Store synthesized PET
        self.fakeB = fakeB.detach()

        return self.genLoss

    def discriminator_training_step(self, imgA, imgB):
        """Update PET discriminator."""

        x = imgA
        y = imgB

        # Generate synthetic PET without generator gradient
        with torch.no_grad():
            fakeB = self.generator(x)

        # Real PET discriminator loss
        predRealB = self.discriminator(y)
        mseRealB = self.get_mse_loss(predRealB, 'real')

        # Generated PET discriminator loss
        predFakeB = self.discriminator(fakeB.detach())
        mseFakeB = self.get_mse_loss(predFakeB, 'fake')

        # L_D = 0.5 * [MSE(D(y),1) + MSE(D(G(x)),0)]
        self.disLoss = 0.5 * (mseRealB + mseFakeB)

        # Progress-bar logging
        self.log('_dis_loss', self.disLoss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False)

        # TensorBoard logging
        self.log('dis_loss', self.disLoss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mse_real_B', mseRealB.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mse_fake_B', mseFakeB.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return self.disLoss

    def training_step(self, batch, batch_idx):
        imgA, imgB = batch['A'], batch['B']

        opt_g, opt_d = self.optimizers()

        # ---------------- Generator update ----------------
        self.toggle_optimizer(opt_g)
        set_requires_grad([self.discriminator], False)

        gen_loss = self.generator_training_step(imgA, imgB)

        opt_g.zero_grad()
        self.manual_backward(gen_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # ---------------- Discriminator update ----------------
        self.toggle_optimizer(opt_d)
        set_requires_grad([self.discriminator], True)

        dis_loss = self.discriminator_training_step(imgA, imgB)

        opt_d.zero_grad()
        self.manual_backward(dis_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        if batch_idx in [5, 105, 205, 305, 505]:
            self.save_image(batch, batch_idx)

    def save_image(self, batch, batch_idx):
        imgA, imgB = batch['A'], batch['B']

        with torch.no_grad():
            fakeB = self.generator(imgA)

        imgA = [self.toImage(f) for f in imgA]
        imgB = [self.toImage(f) for f in imgB]
        fakeB = [self.toImage(f) for f in fakeB]

        for i in range(len(imgA)):
            width = imgA[i].size[0]
            height = imgA[i].size[1]
            new_image = Image.new('RGB', (3 * width, height), (250, 250, 250))
            new_image.paste(imgA[i], (0, 0))
            new_image.paste(imgB[i], (width, 0))
            new_image.paste(fakeB[i], (2 * width, 0))
            image_dir = os.path.join(self.config['model_name'], 'Train Images')
            os.makedirs(image_dir, exist_ok=True)
            image_dir = os.path.join(image_dir, f'epoch_{self.current_epoch:05d}')
            os.makedirs(image_dir, exist_ok=True)
            new_image.save(os.path.join(image_dir, f'{batch_idx:05d}_{i}.png'))

    def toImage(self, x):
        """Convert tensor in range [-1,1] to PIL image."""
        x = x.clone().detach().cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        x = (x + 1) * 127.5
        x = np.clip(x, 0, 255)
        x = x.astype('uint8')
        return Image.fromarray(x)

    def compute_metrics(self, imgB: torch.Tensor, fakeB: torch.Tensor):
        """Compute metrics between synthesized PET and ground-truth PET."""
        with torch.no_grad():
            imgB_255 = convert_to_255(imgB)
            fakeB_255 = convert_to_255(fakeB)

            for metric in self.metric_list:
                self.metric_dict[metric] = calc_metric(self.metric_dict[metric], metric, fakeB_255, imgB_255)
                self.log(f'{metric}_B', self.metric_dict[metric], on_step=True, on_epoch=True, prog_bar=(metric == 'ssim'), logger=True)

    def on_train_epoch_end(self):
        # BUG FIX: with manual optimization, Lightning no longer steps the
        # LR schedulers automatically at epoch end -- it must be done here.
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for sch in schedulers:
            sch.step()

        print('\n')

        for metric in self.metric_list:
            self.metric_dict[metric].reset()

        gc.collect()

        # torch.cuda.empty_cache()
