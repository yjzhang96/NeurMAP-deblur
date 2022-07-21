import argparse
import time
import os
import ipdb
import yaml
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

from torch.utils.data import DataLoader
from data import dataloader_pair
from data import dataloader_unpair


from models import model_semi_double_D_GDaddB_finetune
from models import model_baseline_finetune, model_baseline_finetune_unpair
from models import model_semi_double_D_GOPRO
from utils import utils_new as utils
from tensorboardX import SummaryWriter
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default='./checkpoints/config.yaml')
args = parser.parse_args()

with open(args.config_file,'r') as f:
    config = yaml.load(f)
    for key,value in config.items():
        print('%s:%s'%(key,value))
### make saving dir

if not os.path.exists(config['checkpoints']):
    os.mkdir(config['checkpoints'])
model_save_dir = os.path.join(config['checkpoints'],config['model_name']) 
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
os.system('cp %s %s'%(args.config_file, model_save_dir))



### initialize model
if config['model_class'] == "baseline_finetune_unpair":
    Model = model_baseline_finetune_unpair
    os.system('cp %s %s'%('models/model_baseline_finetune_unpair.py', model_save_dir))

else:
    raise ValueError("Model class [%s] not recognized." % config['model_class'])


### load datasets
if config['dataset_mode'] == 'pair':
    train_dataset = dataloader_pair.BlurryVideo(config, train= True)
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=config['batch_size'],
                                    shuffle = True)
    val_dataset = dataloader_pair.BlurryVideo(config, train= False)
    val_dataloader = DataLoader(val_dataset,
                                    batch_size=config['val']['val_batch_size'],
                                    shuffle = True)
    print("train_dataset:",train_dataset)
    print("val_dataset",val_dataset)                            
elif config['dataset_mode'] == 'unpair':
    train_dataset_unpair = dataloader_unpair.BlurryVideo(config, train= True)
    train_dataloader_unpair = DataLoader(train_dataset_unpair,
                                    batch_size=config['batch_size']//2,
                                    shuffle = True,
                                    num_workers=16)
    val_dataset_unpair = dataloader_unpair.BlurryVideo(config, train= False)
    val_dataloader_unpair = DataLoader(val_dataset_unpair,
                                    batch_size=config['val']['val_batch_size'],
                                    shuffle = True)
    print("train_dataset:",train_dataset_unpair)
    print("val_dataset",val_dataset_unpair)
else:
    raise ValueError("dataset_mode [%s] not recognized." % config['dataset_mode'])

###Create transform to display image from tensor


###Utils
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def write_lr():
    pass


###Initializing VGG16 model for perceptual loss
# vgg16 = torchvision.models.vgg16(pretrained=True)
# vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
# vgg16_conv_4_3.to(device)
# for param in vgg16_conv_4_3.parameters():
# 		param.requires_grad = False



### Initialization
if config['resume_train']:
    if config['which_epoch'] != 'latest':
        config['start_epoch'] = int(config['which_epoch'])
    elif config['which_epoch'] == 'latest':
        assert config['start_epoch'] != 0
else:
    os.system('rm %s/%s/psnr_log.txt'%(config['checkpoints'], config['model_name']))
    # os.system('rm %s/%s/loss_log.txt'%(config['checkpoints'], config['model_name']))
    os.system('rm %s/%s/event*'%(config['checkpoints'], config['model_name']))
# init model
model = Model.DeblurNet(config)
if config['resume_train']:
    model.load(config)

# tensorboard writter
writer = SummaryWriter(model_save_dir)

def display_loss(loss,epoch,tot_epoch,step,step_per_epoch,time):
    loss_writer = ""
    for key, value in loss.items():
        loss_writer += "%s:%.4f\t"%(key,value)
    messege = "epoch[%d/%d],step[%d/%d],time[%.3fs]:%s"%(epoch,tot_epoch,step,step_per_epoch,time,loss_writer)
    print(messege)
    log_name = os.path.join(config['checkpoints'],config['model_name'],'loss_log.txt')   
    with open(log_name,'a') as log:
        log.write(messege+'\n')

# validation
# def validation_pair(epoch):
#     t_b_psnr = 0
#     t_s_psnr = 0
#     t_fakeS_reblur_psnr = 0
#     cnt = 0
#     start_time = time.time()
#     print('--------validation begin----------')
#     for index, batch_data in enumerate(val_dataloader):
#         model.set_input(batch_data)
#         reblur_S_psnr, reblur_fS_psnr, sharp_blur = model.test(validation=True)
#         t_b_psnr += reblur_S_psnr
#         t_fakeS_reblur_psnr += reblur_fS_psnr
#         t_s_psnr += sharp_blur
#         cnt += 1
#         if index > 100:
#             break
#     message = 'UnPair-data epoch %s blur PSNR: %.2f \n'%(epoch, t_fakeS_reblur_psnr/cnt)
#     message += 'UnPair-data epoch %s deblur PSNR: %.2f \n'%(epoch, t_s_psnr/cnt)
#     print(message)
#     print('using time %.3f'%(time.time()-start_time))
#     log_name = os.path.join(config['checkpoints'],config['model_name'],'psnr_log.txt')   
#     with open(log_name,'a') as log:
#         log.write(message)
#     return (t_b_psnr/cnt,t_fakeS_reblur_psnr/cnt, t_s_psnr/cnt)

def validation_unpair(epoch):
    t_b_psnr = 0
    t_s_psnr = 0
    t_fakeS_reblur_psnr = 0
    cnt = 0
    start_time = time.time()
    print('--------validation begin----------')
    for index, batch_data in enumerate(val_dataloader_unpair):
        model.set_input(batch_data)
        reblur_S_psnr, reblur_fS_psnr, sharp_blur = model.test(validation=True)
        t_b_psnr += reblur_S_psnr
        t_fakeS_reblur_psnr += reblur_fS_psnr
        t_s_psnr += sharp_blur
        cnt += 1
        if index > 100:
            break
    message = 'Unpair-data epoch %s blur PSNR: %.2f \n'%(epoch, t_fakeS_reblur_psnr/cnt)
    message += 'Unpair-data epoch %s deblur PSNR: %.2f \n'%(epoch, t_s_psnr/cnt)
    print(message)
    print('using time %.3f'%(time.time()-start_time))
    log_name = os.path.join(config['checkpoints'],config['model_name'],'psnr_log.txt')   
    with open(log_name,'a') as log:
        log.write(message)
    return (t_b_psnr/cnt,t_fakeS_reblur_psnr/cnt, t_s_psnr/cnt)

# training
# val_reblur_S_psnr,val_reblur_fS_psnr, val_deblur_psnr = validation_pair(config['start_epoch'])
# writer.add_scalar('UnPairPSNR/deblur', val_deblur_psnr, config['start_epoch'])
# writer.add_scalar('UnPairPSNR/reblur-S', val_reblur_S_psnr, config['start_epoch'])
# writer.add_scalar('UnPairPSNR/reblur-fakeS', val_reblur_fS_psnr, config['start_epoch'])

best_psnr = 0.0
for epoch in range(config['start_epoch'], config['epoch']):
    epoch_start_time = time.time()
    step_per_epoch = len(train_dataloader_unpair)
    # for step, (batch_data1, batch_data2) in enumerate(zip(train_dataloader_gt,train_dataloader_unpair)):
    G_iter = 0
    D_iter = 0
    for step, batch_data in enumerate(train_dataloader_unpair):
        p = float(step + epoch * step_per_epoch) / config['epoch'] / step_per_epoch
        # # training step 2
        time_step1 = time.time()

        model.set_input(batch_data)        
        model.optimize()     
            

        if step%config['display_freq'] == 0:
            #print a sample result in checkpoints/model_name/samples
            loss = model.get_loss()
            time_ave = (time.time() - time_step1)/config['display_freq']
            display_loss(loss,epoch,config['epoch'],step,step_per_epoch,time_ave)

            results = model.get_current_visuals()
            utils.save_train_sample(config, epoch, results)
            
            for key, value in loss.items():
                writer.add_scalar(key,value,step_per_epoch*epoch+step)

    # schedule learning rate
    model.schedule_lr(epoch,config['epoch'])
    model.save('latest')
    print('End of epoch [%d/%d] \t Time Taken: %d sec' % (epoch, config['epoch'], time.time() - epoch_start_time))

    if epoch%config['save_epoch'] == 0:
        model.save(epoch)
    unpair_results, bmap_vis = model.get_tensorboard_images()
    writer.add_image('UnPair/real_B', unpair_results['real_B'],epoch)
    writer.add_image('UnPair/real_S', unpair_results['real_S'],epoch)
    writer.add_image('UnPair/fake_S', unpair_results['fake_S'],epoch)
    writer.add_image('UnPair/fake_B', unpair_results['fake_B'],epoch)
    writer.add_image('UnPair/B_S_offset', bmap_vis['real_B'],epoch)
    writer.add_image('UnPair/real_S_offset', bmap_vis['real_S'],epoch)
    writer.add_image('UnPair/B_fS_offset', bmap_vis['fake_S'],epoch)

    if epoch%config['val_freq'] == 0:
        val_reblur_S_psnr,val_reblur_fS_psnr, val_deblur_psnr  = validation_unpair(epoch)
        writer.add_scalar('UnPairPSNR/deblur', val_deblur_psnr, epoch)
        writer.add_scalar('UnPairPSNR/reblur-S', val_reblur_S_psnr, epoch)
        writer.add_scalar('UnPairPSNR/reblur-fakeS', val_reblur_fS_psnr, epoch)

        if val_deblur_psnr > best_psnr:
            best_psnr = val_deblur_psnr
            model.save('best')

