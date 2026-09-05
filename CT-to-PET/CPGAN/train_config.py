# Training configuration file
# Single-generator paired CT -> PET image translation

##### DO NOT EDIT THESE LINES #####
config = {}
###################################

#### START EDITING FROM HERE ######

# Dataset and output paths
config['data_path'] = r"...\Data"
config['model_name'] = r"...\Results\resnet_gen_9_patchGAN_paired_prepro"
config['resume_ckpt'] = None
config['paired'] = True

# Training parameters
config['batch_size'] = 2
config['n_lin_epoch'] = 100
config['n_dec_epoch'] = 100
config['save_freq'] = 5

# Model architecture
config['gen_model'] = 'resnet_gen_9'        # ['resnet_gen_9', 'resnet_gen_7', 'resnet_gen_3']
config['dis_model'] = 'patchGAN'            # ['patchGAN', 'pixelGAN']

# Loss functions
config['dis_loss'] = 'mse_loss'             # LSGAN discriminator loss
config['gen_loss'] = 'mse_loss'             # LSGAN generator adversarial loss
config['recon_loss'] = 'mae_loss'           # ['mae_loss', 'mse_loss']
config['identity_loss'] = 'mae_loss'        # ['mae_loss', 'mse_loss']

# Loss weighting coefficients
# L_total = lambda_GAN * L_GAN + lambda_recon * L_recon + lambda_identity * L_identity
# This combination (1 / 100 / 10) is the one that achieved the highest
# mean SSIM during model selection.
config['lambda_gan'] = 1.0
config['lambda_recon'] = 100.0
config['lambda_identity'] = 10.0

# Optimizers
config['gen_opt'] = 'adam'
config['dis_opt'] = 'adam'
config['gen_lr'] = 2e-4
config['dis_lr'] = 2e-4

# Metrics
config['metrics'] = ['fid', 'psnr', 'ssim']
