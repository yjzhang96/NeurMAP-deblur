import argparse
import time
from unittest import TestCase
import yaml
import os
import ipdb
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import dataloader_single
from shutil import rmtree

# from models import model_reblur_gan_lr
import models.networks as networks
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='/hdd/cycle_datasets/Synthetic_blur/trainB')
parser.add_argument("--result_dir", type=str, default='/hdd/deblur_datasets/Synthetic_blur/Gao_quad_small')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gpu", type=str, default='0')
args = parser.parse_args()


### make saving dir
# import ipdb; ipdb.set_trace()
if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

# test
### initialize model

gpu_list = args.gpu.split(',')
gpu_ids = []
for i in gpu_list:
    gpu_ids.append(int(i))

blurnet = networks.define_blur(offset_num=25,gpu_ids=gpu_ids)

### load datasets

dataset = dataloader_single.SingleDataset(args, train= False)
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle = False)                            
print(dataset)


print('--------blur generation begin----------')
for index, batch_data in enumerate(dataloader):
    # if index%2 == 0:
    # Padding in case images are not multiples of 8
    input = batch_data['input']
    input = input.cuda()
    noise = torch.abs(torch.rand(4)) * 0.5
    B, C, H,W = input.shape
    ones_map = torch.ones((B,2,H,W))
    mmap = torch.cat([ones_map * noise[:2].view(1,-1,1,1), -(ones_map * noise[2:].view(1,-1,1,1))], dim=1)
    print('noise',noise)
    ## generate fake quadratic motion map
    blur_img, _ = blurnet(input, mmap.cuda())
    ## save img
    print(blur_img.shape)
    dir, input_name = os.path.split(batch_data['img_path'][0])
    input_name = input_name.split('.')[0][:-6]
    img_path = os.path.join(args.result_dir, '%s_blurryimg.png'%input_name)
    blur_img_np = utils.tensor2im(blur_img)
    utils.save_image(blur_img_np[0], img_path)
    print('generate blurry img:%s'%img_path)