import argparse
import time
from unittest import TestCase
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
from data import dataloader_unpair
from shutil import rmtree

# from models import model_reblur_gan_lr
from models import model_unpair_double_D
from models import model_semi_double_D_GDaddB_finetune
from utils import utils
from models import model_baseline_finetune, model_baseline_finetune_quad, model_baseline_finetune_unpair
from models import model_semi_double_D_GOPRO
            
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
    config['resume_train'] = False
    config['which_epoch'] = config['test']['which_epoch']
    if config['test']['verbose']:
        config['load_only_G'] = False
    utils.print_config(config)

### make saving dir
# import ipdb; ipdb.set_trace()
test_config = config['test']
if not os.path.exists(test_config['result_dir']):
    os.makedirs(test_config['result_dir'],exist_ok=True)
if test_config['save_dir']:
    image_save_dir = os.path.join(test_config['result_dir'],test_config['save_dir']) 
else:
    image_save_dir = os.path.join(test_config['result_dir'],config['model_name']) 

if not os.path.exists(image_save_dir):
    os.mkdir(image_save_dir)

# test
### initialize model



if config['model_class'] == "Semi_doubleD_addB_finetune":
    Model = model_semi_double_D_GDaddB_finetune
elif config['model_class'] == "GOPRO_deblur":
    Model = model_semi_double_D_GOPRO
elif config['model_class'] == "Baseline_finetune":
    Model = model_baseline_finetune
elif config['model_class'] == "Baseline_finetune_quad":
    Model = model_baseline_finetune_quad
elif config['model_class'] == "Baseline_finetune_unpair":
    Model = model_baseline_finetune_unpair
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

t_deblur_psnr = 0
t_s_reblur_psnr = 0
t_fakeS_reblur_psnr = 0
cnt = 0
model.net_G.eval()
model.net_D.eval()
record_file = os.path.join(test_config['result_dir'],test_config['save_dir'],'PSNR.txt')
start_time = time.time()
print('--------testing begin----------')
for index, batch_data in enumerate(test_dataloader):
    # if index%2 == 0:
    # Padding in case images are not multiples of 8
    input = batch_data['B']
    target = batch_data['S']
    if 'RealBlur' in test_config['deblur_videos'] or 'Synthetic' in test_config['deblur_videos']:
        factor = 16
        h,w = input.shape[2], input.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input, (0,padw,0,padh), 'reflect')
        target_ = F.pad(target, (0,padw,0,padh), 'reflect')
        batch_data['B'],batch_data['S'] = input_, target_
        
    start_time_i = time.time()
    model.set_input(batch_data)
    # psnr = model.test(validation=True)
    psnr = model.test_multi_inf(batch_data)
    
    results = model.get_current_visuals()
    image_path = model.get_image_path()
    if 'RealBlur' in test_config['deblur_videos'] or 'Synthetic' in test_config['deblur_videos']:
        results['fake_S'] = results['fake_S'][:,:h,:w,:]
        results['real_S'] = results['real_S'][:,:h,:w,:]
        results['real_B'] = results['real_B'][:,:h,:w,:]
        results['fake_B'] = results['fake_B'][:,:h,:w,:]
    utils.save_test_images(config, image_save_dir, results, image_path)
    if test_config['verbose']:
        B_path = image_path['B_path'][0]
        others, B_name = os.path.split(B_path)
        if os.path.split(others)[-1] == 'blur':
            video_name = others.split('/')[-2]
        else:
            video_name = os.path.split(others)[-1]
        video_dir = os.path.join(image_save_dir, video_name)
        frame_index_B = B_name.split('.')[0]
        # import ipdb;ipdb.set_trace()
        save_path = os.path.join(video_dir,"%s_fake_S_mmap.png"%(frame_index_B))
        mmap_fake_S = model.get_mmap(model.fake_S)
        print("fake_S mean:",torch.abs(mmap_fake_S).mean())
        utils.save_heat_mmap(mmap_fake_S, save_path)

        save_path = os.path.join(video_dir,"%s_real_S_mmap.png"%(frame_index_B))
        mmap_real_S = model.get_mmap(model.real_S)
        print("real_S mean:",torch.abs(mmap_real_S).mean())
        utils.save_heat_mmap(mmap_real_S, save_path)

        save_path = os.path.join(video_dir,"%s_real_B_mmap.png"%(frame_index_B))
        mmap_real_B = model.get_mmap(model.real_B)
        print("real_B mean:",torch.abs(mmap_real_B).mean())
        utils.save_heat_mmap(mmap_real_B, save_path)

        save_path = os.path.join(video_dir,"%s_Reblur_map.png"%(frame_index_B))
        reblur_map = mmap_real_B - mmap_fake_S
        utils.save_heat_mmap(reblur_map, save_path)

    reblur_S_psnr, reblur_fS_psnr, sharp_blur = psnr
    print('[time:%.3f]processing %s PSNR: %.2f'%(time.time()-start_time_i, image_path['B_path'],sharp_blur))
    t_deblur_psnr += sharp_blur
    t_s_reblur_psnr += reblur_S_psnr
    t_fakeS_reblur_psnr += reblur_fS_psnr
    cnt += 1

        
if test_config['dataset_mode'] == 'pair':
    message = 'test %s model on %s PSNR: %.2f'%(config['model_name'], config['test']['blur_videos'], t_deblur_psnr/cnt)
elif test_config['dataset_mode'] == 'unpair':
    message = 'test %s model on %s PSNR: %.2f'%(config['model_name'], config['test']['real_blur_videos'], t_deblur_psnr/cnt)
print(message)
write_txt(record_file, message)
print('using time %.3f'%(time.time()-start_time))
# log_name = os.path.join(image_save_dir,'psnr_log.txt')   
# with open(log_name,'a') as log:
#     log.write(message+'\n')
