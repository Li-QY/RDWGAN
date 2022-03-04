import math
import os
import os.path as osp
import sys

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import yaml
from apex import amp
from addict import Dict
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tensorboard.plugins.mesh import summary as mesh_summary
from libs.modules.loss import AdversarialLoss, PixelLoss, ChamferLoss

# ,PerceptualLoss , TVLoss, ChamferLoss, init_weights
from libs.datasets.mpo_crop import Augmentation
from libs.modules.srgan_mask_skip import Discriminator, Generator
from libs.modules.ssim import SSIM
from utils_aug import (
    evaluator,
    get_device,
    prepare_dataloader,
    set_requires_grad,
    val_calcu,
    _sample_topleft,
)


def scale(imgs):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (imgs + 1.0) / 2.0


def umiscale(imgs):
    return imgs * 2.0 - 1.0


if __name__ == "__main__":
    # Set benchmark
    torch.backends.cudnn.benchmark = True

    # Load a yaml configuration file
    config = Dict(yaml.load(open(sys.argv[1]), Loader=yaml.SafeLoader))
    device = get_device(config.cuda)

    # Dataset
    modal = config.modal[0]
    train_loader, val_loader = prepare_dataloader(config)
    augmentation_T = Augmentation(device, config.train)
    augmentation_V = Augmentation(device, config.val)

    # Interpolation
    interp_factor = config.train.interp_factor
    interp_mode = config.train.interp_mode

    # Model setup
    n_ch = config.train.n_ch
    G = Generator(n_ch, n_ch, interp_factor)

    if config.train.gen_init is not None:
        print("Init:", config.train.gen_init)
        state_dict = torch.load(config.train.gen_init)
        G.load_state_dict(state_dict)
    # G.apply(init_weights)
    G.to(device)
    D = Discriminator(n_ch)
    if config.train.dis_init is not None:
        print("Init:", config.train.dis_init)
        state_dict = torch.load(config.train.dis_init)
        D.load_state_dict(state_dict)
    D.to(device)

    # Loss for training
    criterion_adv = AdversarialLoss("bce").to(device)
    # criterion_vgg = PerceptualLoss("l2").to(device)
    criterion_pix = PixelLoss("l1").to(device)
    # criterion_var = TVLoss().to(device)
    criterion_cham = ChamferLoss().to(device)

    # precalcu
    crop_val = _sample_topleft(config.val)
    vrange = val_calcu(crop_val)
    colors_tensor = torch.zeros([384, 384])
    colors_tensor = colors_tensor.view(-1)
    colors_tensor = colors_tensor[None, ...]
    colors_tensor = torch.cat((colors_tensor, colors_tensor, colors_tensor), dim=1)
    colors_tensor = colors_tensor[None, ...]

    # Optimizer
    optim_G = optim.Adam(G.parameters(), lr=config.train.lr, betas=(0.9, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=config.train.lr, betas=(0.9, 0.999))

    if config.train.opG_init is not None:
        print("Init:", config.train.opG_init)
        state_dict = torch.load(config.train.opG_init)
        optim_G.load_state_dict(state_dict)
    if config.train.opD_init is not None:
        print("Init:", config.train.opD_init)
        state_dict = torch.load(config.train.opD_init)
        optim_D.load_state_dict(state_dict)

    # Apex
    [G, D], [optim_G, optim_D] = amp.initialize(
        [G, D], [optim_G, optim_D], opt_level="O0", num_losses=2, verbosity=0,
    )

    scheduler_G = MultiStepLR(optim_G, config.train.lr_steps, config.train.lr_decay)
    scheduler_D = MultiStepLR(optim_D, config.train.lr_steps, config.train.lr_decay)

    # Experiemtn ID
    exp_id = config.experiment_id

    # Print necessary information
    print("Mask: No | Srgan | Loss: Pixel*1, Adver*1e-3| Augmentation")

    # Tensorboard
    writer = SummaryWriter("runs/" + exp_id)
    os.makedirs(osp.join("models", exp_id), exist_ok=True)
    os.makedirs(osp.join("optimizer", exp_id), exist_ok=True)

    print("Experiment ID:", exp_id)

    # Adversarial training
    n_epoch = math.ceil(config.train.n_iter / len(train_loader))
    for epoch in tqdm(range(1, n_epoch + 1), desc="Epoch"):
        # Training
        G.train()
        for iteration, imgs_HR in tqdm(
            enumerate(train_loader, 1),
            desc="Training/Iteration",
            total=len(train_loader),
            leave=False,
        ):
            crop_start = _sample_topleft(config.train)
            sh, sw = crop_start
            mask_HR = imgs_HR["mask"]
            imgs_HR = imgs_HR["depth"]
            flip = random.randint(0, 1)
            imgs_HR, imgs_LR = augmentation_T(imgs_HR, "depth", flip, crop_start)
            mask_HR, mask_LR = augmentation_T(mask_HR, "mask", flip, crop_start)

            # Generate fake images
            imgs_SR = G(scale(imgs_LR), mask_LR)

            # Update the discriminator
            set_requires_grad(D, True)
            optim_D.zero_grad()

            pred_fake = D(imgs_SR.detach())
            pred_real = D(imgs_HR)
            loss_D = criterion_adv(pred_fake, 0.0)
            loss_D += criterion_adv(pred_real, 1.0)
            loss_D /= 2.0
            with amp.scale_loss(loss_D, optim_D, loss_id=0) as loss_D_scaled:
                loss_D_scaled.backward()
            optim_D.step()

            # Update the generator
            set_requires_grad(D, False)
            optim_G.zero_grad()

            pred_fake = D(imgs_SR)
            loss_adv = criterion_adv(pred_fake, 1.0)
            loss_pix = criterion_pix(imgs_SR, imgs_HR)
            loss_cham = criterion_cham(imgs_SR, imgs_HR, crop_start)
            loss_G = loss_pix
            loss_G += 1e-3 * loss_adv
            loss_G += 1e2 * loss_cham
            with amp.scale_loss(loss_G, optim_G, loss_id=1) as loss_G_scaled:
                loss_G_scaled.backward()
            optim_G.step()

            step = (epoch - 1) * len(train_loader) + iteration
            writer.add_scalar("Loss/Discriminator/Adversarial", loss_D.item(), step)
            writer.add_scalar("Loss/Generator/Adversarial", loss_adv.item(), step)
            writer.add_scalar("Loss/Generator/Image", loss_pix.item(), step)
            writer.add_scalar("Loss/Generator/cham", loss_cham.item(), step)
            writer.add_scalar("Loss/Generator/Total", loss_G.item(), step)

            for i, o in enumerate(optim_G.param_groups):
                writer.add_scalar("LR/Generator/group_{}".format(i), o["lr"], step)
            for i, o in enumerate(optim_D.param_groups):
                writer.add_scalar("LR/Discriminatorgroup_{}".format(i), o["lr"], step)

            scheduler_G.step()
            scheduler_D.step()

        # Validation
        if epoch % config.train.freq_save == 0:
            mse, ssim, psnr, summary, vertices_tensor = evaluator(
                val_loader, G, device, config, step, augmentation_V, vrange
            )
            writer.add_mesh(
                "Point_clouds",
                vertices=vertices_tensor,
                colors=colors_tensor,
                global_step=step,
            )
            writer.add_images("Results", summary, step)
            writer.add_scalar("Score/MSE", mse, step)
            writer.add_scalar("Score/SSIM", ssim, step)
            writer.add_scalar("Score/PSNR", psnr, step)

        if epoch % config.train.freq_save == 0:
            torch.save(
                G.state_dict(),
                osp.join("models", exp_id, "G_epoch_{:05d}.pth".format(epoch)),
            )
            torch.save(
                D.state_dict(),
                osp.join("models", exp_id, "D_epoch_{:05d}.pth".format(epoch)),
            )
            torch.save(
                optim_D.state_dict(),
                osp.join("optimizer", exp_id, "D_epoch_{:05d}.pth".format(epoch)),
            )
            torch.save(
                optim_G.state_dict(),
                osp.join("optimizer", exp_id, "G_epoch_{:05d}.pth".format(epoch)),
            )

