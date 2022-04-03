import argparse
import time
import yaml
import os
import ipdb
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from data import dataloader_pair
from data import dataloader_align_new
from data import dataloader_unpair
from shutil import rmtree
from glob import glob
import scipy.io as sio 
import numpy as np 
# from models import model_reblur_gan_lr
from models import model_semi_double_cGAN
from models import model_unpair_double_D
from models import model_semi_double_D
from models import model_semi_double_D_GDaddB_finetune
from utils import utils_new as utils

            
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default='./checkpoints/config.yaml')
config = parser.parse_args()


def write_txt(file_name, line):
    with open(file_name,'a') as log:
        log.write(line+'\n')
    print(line)


with open(config.config_file,'r') as f:
    config = yaml.load(f)
    config['is_training'] = False
for key,value in config.items():
    print('%s:%s'%(key,value))
### make saving dir
# import ipdb; ipdb.set_trace()
test_config = config['test']
if not os.path.exists(test_config['result_dir']):
    os.mkdir(test_config['result_dir'])
if test_config['save_dir']:
    image_save_dir = os.path.join(test_config['result_dir'],test_config['save_dir']) 
else:
    image_save_dir = os.path.join(test_config['result_dir'],config['model_name']) 

if not os.path.exists(image_save_dir):
    os.mkdir(image_save_dir)

# test
### initialize model


if config['model_class'] == "CGAN_double_semi":
    Model = model_semi_double_cGAN
elif config['model_class'] == "double_D":
    Model = model_unpair_double_D
elif config['model_class'] == "Semi_blurD":
    Model = model_semi_double_D
elif config['model_class'] == "Semi_doubleD_addB_finetune":
    Model = model_semi_double_D_GDaddB_finetune
else:
    raise ValueError("Model class [%s] not recognized." % config['model_class'])

model = Model.DeblurNet(config)
model.load(config)

### load datasets

if test_config['dataset_mode'] == 'pair':
    test_dataset = dataloader_pair.BlurryVideo(config, train= False)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=test_config['test_batch_size'],
                                    shuffle = False)                            
elif test_config['dataset_mode'] == 'unpair':
    test_dataset = dataloader_unpair.BlurryVideo(config, train= False)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=test_config['test_batch_size'],
                                    shuffle = False)  
print(test_dataset)

t_deblur_epe = 0

cnt = 0
record_file = os.path.join(test_config['result_dir'],test_config['save_dir'],'epe.txt')
gtfile_pattern = '/home/yjz/datasets/Synthetic_blur/test_small/*_mfmap.mat'
mmap_gt_files = sorted(glob(gtfile_pattern))

def cal_epe(reblur_map, gt_map):
    reblur_map = torch.nn.functional.interpolate(reblur_map,size=(gt_map.shape[0],gt_map.shape[1]))
    reblur_map = reblur_map.cpu().numpy()
    reblur_map = np.transpose(np.squeeze(reblur_map), (1,2,0))
    reblur_map = reblur_map * 20
    index = np.where(reblur_map[...,1] < 0)
    reblur_map[index] = -reblur_map[index]  
    reblur_map = np.flip(reblur_map, axis=-1)
    reblur_map = reblur_map[...,]
    # import ipdb; ipdb.set_trace()
    
    error = np.mean(np.square( reblur_map - gt_map))
    error_reverse = np.mean(np.square( -reblur_map - gt_map))
    error_final = min(error,error_reverse)
    print('EPE:', error_final)
    return error_final

start_time = time.time()
print('--------testing begin----------')
for index, batch_data in enumerate(test_dataloader):
    with torch.no_grad():
        # if index%2 == 0:
        start_time_i = time.time()
        model.set_input(batch_data)
        model.test(validation=False)
        
        results = model.get_current_visuals()
        image_path = model.get_image_path()
        utils.save_test_images(config, image_save_dir, results, image_path)
            
        B_path = image_path['B_path'][0]
        others, B_name = os.path.split(B_path)
        if os.path.split(others)[-1] == 'blur':
            video_name = others.split('/')[-2]
        else:
            video_name = os.path.split(others)[-1]
        video_dir = os.path.join(image_save_dir,'vis_mmap', video_name)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir,exist_ok=True)
        frame_index_B = B_name.split('.')[0]
        # import ipdb;ipdb.set_trace()
        bmap_fake_S = model.get_bmap(model.fake_S)

        bmap_real_B = model.get_bmap(model.real_B)

        save_path = os.path.join(video_dir,"%s_Reblur_map.png"%(frame_index_B))
        reblur_map = bmap_real_B - bmap_fake_S
        utils.save_heat_bmap(reblur_map, save_path)

        gt_mmap = sio.loadmat(mmap_gt_files[index])['mfmap']
        t_deblur_epe += cal_epe(reblur_map, gt_mmap)
        
        print('[time:%.3f]processing %s '%(time.time()-start_time_i,image_path['B_path']))

        cnt += 1

        
if test_config['dataset_mode'] == 'pair':
    message = 'test %s model on %s epe: %.2f'%(config['model_name'], config['test']['blur_videos'], t_deblur_epe/cnt)
elif test_config['dataset_mode'] == 'unpair':
    message = 'test %s model on %s epe: %.2f'%(config['model_name'], config['test']['real_blur_videos'], t_deblur_epe/cnt)
print(message)
write_txt(record_file, message)
print('using time %.3f'%(time.time()-start_time))

