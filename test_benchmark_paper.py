import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator
# from model_ori import Generator
from model_wgan_Dense import Generator_RDN

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_ours_name', default='netG_epoch_4_481.pth', type=str, help='SDRGAN model epoch name')
parser.add_argument('--model_bl_name', default='netG_epoch_4_481_bl.pth', type=str, help='BL model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
OURS_MODEL_NAME = opt.model_ours_name
BL_MODEL_NAME = opt.model_bl_name
use_cuda = False
exp_id = 'paper'
epoch = '-481'

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}
model_ours = Generator_RDN(4,3,64,4,5)
model_bl = Generator(UPSCALE_FACTOR)
if use_cuda:
    model_bl.cuda()
    model_ours.cuda()
    model_ours.load_state_dict(torch.load('epochs/' + OURS_MODEL_NAME))
    model_bl.load_state_dict(torch.load('epochs/' + BL_MODEL_NAME))

else:
    model_ours.load_state_dict(torch.load('epochs/' + OURS_MODEL_NAME, map_location="cpu"))
    model_bl.load_state_dict(torch.load('epochs/' + BL_MODEL_NAME, map_location="cpu"))
#,map_location='cpu'

test_set = TestDatasetFromFolder('/Users/li/Downloads/data/test', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/' + exp_id + epoch + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

for image_name, lr_image, hr_restore_img, hr_image, hr_restore_nea in test_bar:
    image_name = image_name[0]
    # lr_image = Variable(lr_image, volatile=True)
    # hr_image = Variable(hr_image, volatile=True)
    if use_cuda:
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    bl_image = model_bl(lr_image)
    ours_image = model_ours(lr_image)

    mse_bl = ((hr_image - bl_image) ** 2).data.mean()
    mse_ours = ((hr_image - ours_image) ** 2).data.mean()

    psnr_bl = 10 * log10(1 / mse_bl)
    psnr_ours = 10 * log10(1 / mse_ours)

    ssim_bl = pytorch_ssim.ssim(bl_image, hr_image).item()
    ssim_ours = pytorch_ssim.ssim(ours_image, hr_image).item()

    test_images = torch.stack(
        [ display_transform()(hr_image.data.cpu().squeeze(0)),display_transform()(hr_restore_img.squeeze(0))
        ,display_transform()(hr_restore_nea.squeeze(0)),display_transform()(bl_image.data.cpu().squeeze(0)),
        display_transform()(ours_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=5, padding=5)
    utils.save_image(image, out_path + image_name.split('.')[0] + 'BL_psnr_%.4fOurs_psnr_%.4f.' % (psnr_bl, psnr_ours) +
                     'BL_ssim_%.4fOurs_ssim_%.4f.' % (ssim_bl, ssim_ours) + image_name.split('.')[-1], padding=5)

    # save psnr\ssim
    # results[image_name.split('_')[0]]['psnr'].append(psnr)
    # results[image_name.split('_')[0]]['ssim'].append(ssim)

# out_path = 'statistics/'
# saved_results = {'psnr': [], 'ssim': []}
# for item in results.values():
#     psnr = np.array(item['psnr'])
#     ssim = np.array(item['ssim'])
#     if (len(psnr) == 0) or (len(ssim) == 0):
#         psnr = 'No data'
#         ssim = 'No data'
#     else:
#         psnr = psnr.mean()
#         ssim = ssim.mean()
#     saved_results['psnr'].append(psnr)
#     saved_results['ssim'].append(ssim)

# data_frame = pd.DataFrame(saved_results, results.keys())
# data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_' + exp_id + epoch + '.csv',index_label='DataSet')
