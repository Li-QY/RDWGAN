import argparse
import time
from numpy.lib.type_check import imag

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model_wgan_Dense import Generator_RDN
from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name',default="/Users/li/Downloads/data/test/SRF_4/data/BSD100_001.png",type=str, help='test low resolution image name')
parser.add_argument('--model_ours_name', default='netG_epoch_4_481.pth', type=str, help='generator model epoch name')
parser.add_argument('--model_bl_name', default='netG_epoch_4_481_bl.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
OURS_MODEL_NAME = opt.model_ours_name
BL_MODEL_NAME = opt.model_bl_name

model_ours = Generator_RDN(4,3,64,4,5).eval()
model_bl = Generator(4).eval()
if TEST_MODE:
    model_bl.cuda()
    model_ours.cuda()
    model_ours.load_state_dict(torch.load('epochs/' + OURS_MODEL_NAME))
    model_ours.load_state_dict(torch.load('epochs/' + BL_MODEL_NAME))

else:
    model_ours.load_state_dict(torch.load('epochs/' + OURS_MODEL_NAME, map_location="cpu"))
    model_ours.load_state_dict(torch.load('epochs/' + BL_MODEL_NAME, map_location="cpu"))

image = Image.open(IMAGE_NAME)
with torch.no_grad():
    image = ToTensor()(image).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    # start = time.clock()
    out_ours = model_ours(image)
    out_bl = model_bl(image)
# elapsed = (time.clock() - start)
# print('cost' + str(elapsed) + 's')
# out_img = ToPILImage()(out[0].data.cpu())
# out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_OutImage.png')
