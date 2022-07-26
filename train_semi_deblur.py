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


from models import model_baseline_finetune_diff, model_semi_double_D_GDaddB_finetune
from models import model_baseline_finetune, model_baseline_finetune_unpair

from utils import utils
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

### assign GPU
# if config['gpu'] >= 0:
#     device = torch.device('cuda:{}'.format(self.gpu[0]))
# else:
#     device = torch.device("cpu")

### initialize model
if config['model_class'] == "Semi_doubleD_addB_finetune":
    Model = model_semi_double_D_GDaddB_finetune
    os.system('cp %s %s'%('models/model_semi_double_D_GDaddB_finetune.py', model_save_dir))
elif config['model_class'] == "baseline_finetune":
    Model = model_baseline_finetune
    os.system('cp %s %s'%('models/model_baseline_finetune.py', model_save_dir))
elif config['model_class'] == "Diff_GT":
    Model = model_baseline_finetune_diff
    os.system('cp %s %s'%('models/model_baseline_finetune_diff.py', model_save_dir))
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
elif config['dataset_mode'] == 'mix':
    train_dataset = dataloader_pair.BlurryVideo(config, train= True)
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=config['batch_size']//2,
                                    shuffle = True,
                                    num_workers=16)
    train_dataset_unpair = dataloader_unpair.BlurryVideo(config, train= True)
    train_dataloader_unpair = DataLoader(train_dataset_unpair,
                                    batch_size=config['batch_size']//2,
                                    shuffle = True,
                                    num_workers=16)
    val_dataset = dataloader_pair.BlurryVideo(config, train= False)
    val_dataloader = DataLoader(val_dataset,
                                    batch_size=config['val']['val_batch_size'],
                                    shuffle = True)
    val_dataset_unpair = dataloader_unpair.BlurryVideo(config, train= False)
    val_dataloader_unpair = DataLoader(val_dataset_unpair,
                                    batch_size=config['val']['val_batch_size'],
                                    shuffle = True)
else:
    raise ValueError("dataset_mode [%s] not recognized." % config['dataset_mode'])
print("train_dataset:",train_dataset)
print("train_dataset_unpair:",train_dataset_unpair)
print("val_dataset",val_dataset)
print("val_dataset_unpair",val_dataset_unpair)
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
# if config['resume_train']:
#     model.load(config)

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
def validation_pair(epoch):
    t_b_psnr = 0
    t_s_psnr = 0
    t_fakeS_reblur_psnr = 0
    cnt = 0
    start_time = time.time()
    print('--------validation begin----------')
    for index, batch_data in enumerate(val_dataloader):
        model.set_input(batch_data)
        reblur_S_psnr, reblur_fS_psnr, sharp_blur = model.test(validation=True)
        t_b_psnr += reblur_S_psnr
        t_fakeS_reblur_psnr += reblur_fS_psnr
        t_s_psnr += sharp_blur
        cnt += 1
        if index > 100:
            break
    message = 'Pair-data epoch %s blur PSNR: %.2f \n'%(epoch, t_fakeS_reblur_psnr/cnt)
    message += 'Pair-data epoch %s deblur PSNR: %.2f \n'%(epoch, t_s_psnr/cnt)
    print(message)
    print('using time %.3f'%(time.time()-start_time))
    log_name = os.path.join(config['checkpoints'],config['model_name'],'psnr_log.txt')   
    with open(log_name,'a') as log:
        log.write(message)
    return (t_b_psnr/cnt,t_fakeS_reblur_psnr/cnt, t_s_psnr/cnt)

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
#val_reblur_S_psnr,val_reblur_fS_psnr, val_deblur_psnr = validation_pair(config['start_epoch'])
#writer.add_scalar('PairPSNR/deblur', val_deblur_psnr, config['start_epoch'])
#writer.add_scalar('PairPSNR/reblur-S', val_reblur_S_psnr, config['start_epoch'])
#writer.add_scalar('PairPSNR/reblur-fakeS', val_reblur_fS_psnr, config['start_epoch'])
# val_reblur_S_psnr,val_reblur_fS_psnr, val_deblur_psnr = validation_unpair(config['start_epoch'])
# writer.add_scalar('UnpairPSNR/deblur', val_deblur_psnr, config['start_epoch'])
# writer.add_scalar('UnpairPSNR/reblur-S', val_reblur_S_psnr, config['start_epoch'])
# writer.add_scalar('UnpairPSNR/reblur-fakeS', val_reblur_fS_psnr, config['start_epoch'])

best_psnr = 0.0
for epoch in range(config['start_epoch'], config['epoch']):
    epoch_start_time = time.time()
    step_per_epoch = min(len(train_dataloader), len(train_dataloader_unpair))
    # for step, (batch_data1, batch_data2) in enumerate(zip(train_dataloader_gt,train_dataloader_unpair)):
    G_iter = 0
    D_iter = 0
    for step, (batch_data1, batch_data_unpair) in enumerate(zip(train_dataloader,train_dataloader_unpair)):
        p = float(step + epoch * step_per_epoch) / config['epoch'] / step_per_epoch
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # # training step 2
        time_step1 = time.time()
        batch_cat = {}
        batch_cat['B'] = torch.cat((batch_data1['B'],batch_data_unpair['B']),dim=0)
        batch_cat['S'] = torch.cat((batch_data1['S'],batch_data_unpair['S']),dim=0)
        batch_cat['B_path'] = batch_data1['B_path'] + batch_data_unpair['B_path']
        batch_cat['gt'] = torch.cat((batch_data1['gt'],batch_data_unpair['gt']),dim=0)
        model.set_input(batch_cat)        
        if config['model_class'] == "Semi_doubleD_addB_DomAda":
            model.optimize(alpha=alpha)
        else:
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

    if epoch != 0 and epoch%config['save_epoch'] == 0:
        model.save(epoch)
    paired_results, unpaired_results, mmap_vis = model.get_tensorboard_images()
    writer.add_image('Pair/real_B', paired_results['real_B'],epoch)
    writer.add_image('Pair/real_S', paired_results['real_S'],epoch)
    writer.add_image('Pair/fake_S', paired_results['fake_S'],epoch)
    writer.add_image('Pair/fake_B', paired_results['fake_B'],epoch)
    writer.add_image('UnPair/real_B', unpaired_results['real_B'],epoch)
    writer.add_image('UnPair/real_S', unpaired_results['real_S'],epoch)
    writer.add_image('UnPair/fake_S', unpaired_results['fake_S'],epoch)
    writer.add_image('UnPair/fake_B', unpaired_results['fake_B'],epoch)
    writer.add_image('UnPair/B_S_offset', mmap_vis['real_B'],epoch)
    writer.add_image('UnPair/real_S_offset', mmap_vis['real_S'],epoch)
    writer.add_image('UnPair/B_fS_offset', mmap_vis['fake_S'],epoch)

    if epoch%config['val_freq'] == 0:
        val_reblur_S_psnr,val_reblur_fS_psnr, val_deblur_psnr  = validation_pair(epoch)
        writer.add_scalar('PairPSNR/deblur', val_deblur_psnr, epoch)
        writer.add_scalar('PairPSNR/reblur-S', val_reblur_S_psnr, epoch)
        writer.add_scalar('PairPSNR/reblur-fakeS', val_reblur_fS_psnr, epoch)

        val_reblur_S_psnr,val_reblur_fS_psnr, val_deblur_psnr  = validation_unpair(epoch)
        writer.add_scalar('UnpairPSNR/deblur', val_deblur_psnr, epoch)
        writer.add_scalar('UnpairPSNR/reblur-S', val_reblur_S_psnr, epoch)
        writer.add_scalar('UnpairPSNR/reblur-fakeS', val_reblur_fS_psnr, epoch)
        if val_deblur_psnr > best_psnr:
            best_psnr = val_deblur_psnr
            model.save('best')

