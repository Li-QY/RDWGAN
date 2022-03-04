import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, set_requires_grad
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=500, type=int, help='train epoch number')
parser.add_argument('--growthRate', default=64, type=int, help='Denselayer growth rate')
parser.add_argument('--nDenselayer', default=4, type=int, help='Number of Denselayers')
parser.add_argument('--nBlock', default=5, type=int, help='Number of Dense Blocks')


if __name__ == '__main__':
    
    torch.backends.cudnn.benchmark = True

    exp_id = "srgan_bl"
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    GROWTHRATE = opt.growthRate
    DENSELAYERS = opt.nDenselayer
    DENSEBLOCKS = opt.nBlock
    CRITIC_ITERS = 5
    check_point = -1
    n_epoch_pretrain = 3
    on_Mac = False

    if on_Mac :
        train_path = '/Users/li/Downloads/data/train_mac/'
        val_path = '/Users/li/Downloads/data/val_mac/'
    else :
        train_path = '/home/mist/VOC2012/train'
        val_path = '/home/mist/VOC2012/val'

    train_set = TrainDatasetFromFolder(train_path, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder(val_path, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()
    mse = nn.MSELoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
        mse.cuda()
        device = True
    else:
        device = False

    # modification 2: Use RMSprop instead of Adam
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    # Tensorboard
    writer = SummaryWriter("runs/" + exp_id)

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epoch"):
        # train_bar = tqdm(train_loader,leave=False)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for iteration, datas in tqdm(enumerate(train_loader, 1),
            desc="Training/Iteration",
            total=len(train_loader),
            leave=False,
        ):
            data, target = datas
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)
            
            set_requires_grad(netD, True)
            optimizerD.zero_grad()
            
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img.detach()).mean()

            # Modification 5: WGAN-gp
            # Modification 3: remove log(adv) 
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            # if iteration % CRITIC_ITERS == 0 :
            set_requires_grad(netD, False)
            optimizerG.zero_grad()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            image_loss, adversarial_loss, perception_loss, tv_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss = image_loss + adversarial_loss+  perception_loss+ tv_loss
            g_loss.backward()

            optimizerG.step()


            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size

            step = (epoch - 1) * len(train_loader) + iteration
            writer.add_scalar("Loss/Discriminator", d_loss.item(), step)
            writer.add_scalar("Loss/Generator", g_loss.item(), step)
            writer.add_scalar("Loss/image_loss", image_loss.item(), step)
            writer.add_scalar("Loss/adversarial_loss", adversarial_loss.item(), step)
            writer.add_scalar("Loss/perception_loss", perception_loss.item(), step)
            writer.add_scalar("Loss/tv_loss", tv_loss.item(), step)

        netG.eval()
        out_path = 'training_results/' + exp_id + 'SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_iteration, datas in tqdm(
            enumerate(val_loader, 1),
            desc="Evaluation/Iteration",
            total=len(val_loader),
            leave=False,
            ):
                val_lr, val_hr_restore, val_hr = datas
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                    val_hr_restore = val_hr_restore.cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                if val_iteration == 1:
                    val_images.extend([val_hr_restore, hr, sr])
            val_images = torch.cat(val_images, dim=3)

        # save model parameters
        out_path = 'epochs/' + exp_id
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        torch.save(netG.state_dict(), out_path + '/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), out_path + '/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        writer.add_images("Results", val_images, step)
        writer.add_scalar("Score/SSIM", valing_results['ssim'], step)
        writer.add_scalar("Score/PSNR", valing_results['psnr'], step)

        out_path = 'statistics/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if epoch % 10 == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + exp_id + '_train_results.csv', index_label='Epoch')
