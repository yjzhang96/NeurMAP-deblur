import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import os
import torch.nn.functional as F
from . import networks
from collections import OrderedDict
from utils import utils
from .losses import get_loss, SSIMLoss
from .schedulers import WarmRestart,LinearDecay
from utils.image_pool import ImagePool
from ipdb import set_trace as stc
import copy

# calculate PSNR
def PSNR(img1, img2):
    MSE = F.mse_loss(img1,img2)
    return 10 * np.log10(1 / MSE.item())
class DeblurNet():
    def __init__(self, config):
        self.config = config
        if config['gpu']:
            self.device = torch.device('cuda:{}'.format(config['gpu'][0]))
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        ### initial model
        self.net_G = networks.define_net_G(config)
        self.n_offset = 15
        self.net_M = networks.define_net_M(config)
        self.net_naturalD = networks.define_natural_D(config)
        self.blur_net = networks.define_blur(input_nc=3, output_nc=3, offset_num=self.n_offset, gpu_ids=config['gpu'])      # deformable

        ###Loss and Optimizer
        self.criterion_reblur = get_loss('reblur')
        self.criterion_lsgan = get_loss('lsgan')
        self.MSE = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.SSIMloss = SSIMLoss('MSSSIM')
        

        ### Image pool
        self.real_pool = ImagePool(50)
        if config['is_training']:
            if config['model']['g_name'] == 'MPRnet':
                load_G_model = config['MPR_model']
                try:
                    self.net_G.module.load_state_dict(torch.load(load_G_model)['state_dict'])
                except:
                    self.net_G.load_state_dict(torch.load(load_G_model)['state_dict'])
                print('--------load model %s success!-------'%load_G_model)
            if config['model']['g_name'] == 'DMPHN':
                if config.get('DMPHN'):
                    load_G_model = config['DMPHN_model']
                    try: 
                        self.net_G.module.load_state_dict(torch.load(load_G_model))
                    except:
                        self.net_G.load_state_dict(torch.load(load_G_model))
                    print('--------load model %s success!-------'%load_G_model)
                    print('-------- teacher model created -------') 
            if config['model']['g_name'] == 'MIMO':
                load_G_model = config['MIMO_model']
                try: 
                    self.net_G.module.load_state_dict(torch.load(load_G_model)['model'])
                except:
                    self.net_G.load_state_dict(torch.load(load_G_model)['model'])
            if config.get('MTR_model', None):
                load_D_model = config['MTR_model']
                try: 
                    self.net_M.module.load_state_dict(torch.load(load_D_model))
                except:
                    self.net_M.load_state_dict(torch.load(load_D_model)) 
  

        if config['is_training']:
            self.optimizer_M = torch.optim.Adam( self.net_M.parameters(), lr=config['train']['lr_M'], betas=(0.9, 0.999) )
            self.optimizer_G = torch.optim.Adam( self.net_G.parameters(), lr=config['train']['lr_G'], betas=(0.9, 0.999) )
            self.optimizer_naturalD = torch.optim.Adam( self.net_naturalD.parameters(), lr=config['train']['lr_D'], betas=(0.9, 0.999) )
            
        if config['resume_train']:
            print("------loading learning rate------")
            self.load(config)
            self.get_current_lr_from_epoch(self.optimizer_G, config['train']['lr_G'], config['start_epoch'], config['epoch'])
            self.get_current_lr_from_epoch(self.optimizer_M, config['train']['lr_D'], config['start_epoch'], config['epoch'])
            self.get_current_lr_from_epoch(self.optimizer_naturalD, config['train']['lr_D'], config['start_epoch'], config['epoch'])

        # for param in self.net_M.parameters():
        #     param.requires_grad = False
        

    def set_input(self,batch_data):
        # self.real_B = batch_data['B'].to(self.device)
        self.real_B = batch_data['B'].cuda()
        if batch_data['gt'][0]:
            self.real_S_exist = True
            self.real_S = batch_data['S'].cuda()
        else:
            self.real_S_exist = False
            self.real_S = batch_data['S'].cuda()
        self.B_path = batch_data['B_path']


    def optimize(self):
        self.fake_S = self.net_G(self.real_B)
        self.update_M()
        if self.config['train']['natural_D']:
            self.update_naturalD()
        else:
            self.loss_natural_D = torch.tensor(0.0)
        self.update_G()



    @staticmethod
    def vec_norm(vec):
        B,C,H,W = vec.shape
        if C == 2:
            # vec_trans = torch.transpose(vec,(0,2,3,1))
            x,y = torch.chunk(vec,2,dim=1)
            
            x_norm = torch.where(y<0,-x,x)
            y_norm = torch.where(y<0,-y,y)
            vec_norm = torch.cat((x_norm,y_norm),dim=1)
        elif C == 4:
            vec_norm = vec
        return vec_norm

    
    def vec_diff(self, vec1, vec2):
        vec1_abs = self.vec_norm(vec1)
        vec2_abs = self.vec_norm(vec2)
        diff = vec1_abs - vec2_abs
        return diff

    def offset_reg_loss(self,mmap):
        tv_loss = self.L1loss(mmap[:,:,:,:-1],mmap[:,:,:,1:]) + \
                        self.L1loss(mmap[:,:,:-1,:],mmap[:,:,1:,:])

        # # regulationl loss
        # lambda_reg = 0.0
        reg_loss = torch.mean(mmap**2)
        return tv_loss + 0.1 * reg_loss

    def update_M(self,warmup=False):
        # import ipdb; ipdb.set_trace()
        B,C,H,W = self.real_B.shape
        lambda_d_S = self.config['train']['lambda_M_sharp']
        lambda_d_fS = self.config['train']['lambda_M_f_sharp']
        lambda_d_B = self.config['train']['lambda_M_blur']
        lambda_d_tv = self.config['train']['lambda_M_tv']
        lambda_d_reblur = self.config['train']['lambda_M_reblur']

        mmap_real_B = self.net_M(self.real_B)
        mmap_real_S = self.net_M(self.real_S)
        fake_S_detach = self.fake_S.detach()
        mmap_fake_S = self.net_M(fake_S_detach)
        
        mmap_real_B_norm = self.vec_norm(mmap_real_B)
        mmap_real_S_norm = self.vec_norm(mmap_real_S)
        mmap_fake_S_norm = self.vec_norm(mmap_fake_S)
        self.mmaps = {'real_B':mmap_real_B_norm,'fake_S':mmap_fake_S_norm,'real_S':mmap_real_S_norm}

        one_tensor = torch.ones_like(mmap_real_B).cuda()
        zero_tensor = torch.zeros_like(mmap_real_B).cuda()
        # add image pool
        self.real_pool.add(mmap_real_S_norm)
        
        ## offset to 0/1
        self.loss_d_sharp = self.L1loss(mmap_real_S_norm, zero_tensor)
        mmap_real_B_norm_detach = mmap_real_B_norm.detach()
        self.loss_d_fake_sharp = self.L1loss(mmap_fake_S_norm, mmap_real_B_norm_detach)
        ## motion map B as large as possible (upper limit)
        self.loss_d_blur = self.L1loss(torch.abs(mmap_real_B_norm), one_tensor)
        ## motion map B as large as possible (no upper limit)
        # self.loss_d_blur = - self.L1loss(torch.abs(mmap_real_B_norm), zero_tensor)
        self.loss_adv_D  = lambda_d_S * self.loss_d_sharp \
                                + lambda_d_fS * self.loss_d_fake_sharp + lambda_d_B * self.loss_d_blur
        

        # loss: Map_fS is smaller than Map_B
        Mag_fake_S = torch.sqrt(mmap_fake_S_norm[:,0,:,:]**2 + mmap_fake_S_norm[:,1,:,:]**2)
        Mag_real_B = torch.sqrt(mmap_real_B_norm[:,0,:,:]**2 + mmap_real_B_norm[:,1,:,:]**2)
        Mag_diff = Mag_fake_S - Mag_real_B
        Mag_gt = torch.where(Mag_diff>0, Mag_diff, torch.zeros_like(Mag_diff))
        self.loss_Mag_gt = torch.mean(Mag_gt)  

        # reblur loss       
        if self.config['train']['relative_reblur']:
            mmap_rB_fS = mmap_real_B_norm - mmap_fake_S_norm
            self.fake_B_from_fS, offsets = self.blur_net(fake_S_detach, mmap_rB_fS)
            # mmap_rB_rS = mmap_real_B - mmap_real_S.detach()
            # self.fake_B_from_S , _ = self.blur_net(self.real_S[:B//2], mmap_rB_rS[:B//2])
        elif self.config['train']['absolute_reblur']:
            self.fake_B, offsets = self.blur_net(fake_S_detach, mmap_real_B_norm)
        

        B,C,H,W = self.real_B.shape
        offsets	= offsets.view(B,15,-1,H,W)
        
        # # spatial tv_loss across one offset
        # lambda_tv = 0.002
        tv_loss = self.L1loss(offsets[:,:,:,:,:-1],offsets[:,:,:,:,1:]) + \
                        self.L1loss(offsets[:,:,:,:-1,:],offsets[:,:,:,1:,:])

        # # regulationl loss
        lambda_reg = 0.0002
        # lambda_reg = 0.0
        reg_loss = torch.mean(offsets[:,:,:,:,:]**2)
        
        # MSE loloss
        loss_MSE_fS = self.MSE(self.fake_B_from_fS,self.real_B)

        # SSIM loss
        lambda_SSIM = 0.1
        ssim_loss_fS = 1 - self.SSIMloss.get_loss(self.fake_B_from_fS,self.real_B)

        self.loss_d_reblur =  lambda_SSIM * ssim_loss_fS + lambda_d_reblur * loss_MSE_fS \
                            + lambda_reg * reg_loss + lambda_d_tv * tv_loss 

        self.loss_total = self.loss_adv_D   + self.loss_d_reblur + self.loss_Mag_gt
        # import ipdb; ipdb.set_trace()
        self.optimizer_M.zero_grad()
        self.loss_total.backward()
        if self.config['model']['g_name'] == 'MPRnet':
            torch.nn.utils.clip_grad_norm_(self.net_M.parameters(), 0.01)

        self.optimizer_M.step()
        
    def update_naturalD(self):
        B,C,H,W = self.real_B.shape

        pred_fake = self.net_naturalD(self.fake_S.detach())
        loss_D_fake = self.criterion_lsgan(pred_fake, False)

        mix_real_batch = torch.cat((self.real_S[0:B:2],self.real_B[1:B:2]),dim=0)
        # import ipdb; ipdb.set_trace()
        if "Synthetic" in self.config['train']['real_blur_videos'] or "RealBlur" in self.config['train']['real_blur_videos']:
            pred_real = self.net_naturalD(self.real_S)
        else:
            # raise ValueError('wrong dataset')
            pred_real = self.net_naturalD(mix_real_batch)

        loss_D_real = self.criterion_lsgan(pred_real, True)
        self.loss_natural_D = loss_D_fake + loss_D_real 
        self.optimizer_naturalD.zero_grad()
        self.loss_natural_D.backward()
        self.optimizer_naturalD.step()
    
    
    def update_G(self,warmup=False):
        B,C,H,W = self.real_B.shape

        lambda_G_fS = self.config['train']['lambda_G_f_sharp']
        lambda_G_idt = self.config['train']['lambda_G_idt']
        lambda_G_tv = self.config['train']['lambda_G_tv']
        lambda_G_teach = self.config['train']['lambda_G_teacher']
        lambda_G_reblur = self.config['train']['lambda_G_reblur']
        lambda_G_global = self.config['train']['lambda_G_global']
        lambda_G_Itv = self.config['train']['lambda_G_Itv']

        mmap_fake_S = self.net_M(self.fake_S)
        mmap_real_B = self.net_M(self.real_B)
        mmap_fake_S_norm = self.vec_norm(mmap_fake_S)
        ### G_fs to zero
        zero_tensor = torch.zeros_like(mmap_real_B).cuda()
        self.loss_adv_G_fS =  self.L1loss(torch.abs(mmap_fake_S_norm), zero_tensor)
        ### G_fs to mean(real_S)  ##relativistic loss
        # real_pool_query = torch.abs(self.real_pool.query())
        # real_pool_mean = torch.mean(real_pool_query,(0,2,3))
        # real_pool_mean = real_pool_mean.view(1,-1,1,1)
        # self.loss_adv_G_fS =  torch.mean((torch.abs(mmap_fake_S_norm) - real_pool_mean)**2)
        
        
        ## G tv loss
        # tv loss for blur map
        self.G_tv_loss = self.L1loss(mmap_fake_S_norm[:,:,:,:-1],mmap_fake_S_norm[:,:,:,1:]) + \
                        self.L1loss(mmap_fake_S_norm[:,:,:-1,:],mmap_fake_S_norm[:,:,1:,:])
        # tv loss for generated image
        self.G_Itv_loss = self.L1loss(self.fake_S[:,:,:,:-1],self.fake_S[:,:,:,1:]) + \
                        self.L1loss(self.fake_S[:,:,:-1,:],self.fake_S[:,:,1:,:])
        
        self.loss_adv_G = lambda_G_fS * self.loss_adv_G_fS 
        # import ipdb; ipdb.set_trace()

        # global G loss
        if self.config['train']['natural_D']:
            pred_fake = self.net_naturalD(self.fake_S)
            self.loss_adv_globalG = self.criterion_lsgan(pred_fake, True)
        else:
            self.loss_adv_globalG = torch.tensor(0.0)

        # reblur loss to train G
        if self.config['train']['relative_reblur']:
            # import ipdb; ipdb.set_trace()
            mmap_real_B_norm = self.vec_norm(mmap_real_B)
            reblur_mmap = mmap_real_B_norm - mmap_fake_S_norm
        elif self.config['train']['absolute_reblur']:
            reblur_mmap = mmap_real_B
        reblur_mmap_detach = reblur_mmap.detach()
        self.fake_B, _ = self.blur_net(self.fake_S, reblur_mmap_detach)
        self.loss_reblur = self.criterion_reblur(self.real_B, self.fake_B)

        # apply content loss to warm start net_G
        if self.config['train']['identical_loss']:
            real_S_idt = self.net_G(self.real_S)
            self.loss_idt = self.L1loss(self.real_S, real_S_idt)            
        else:
            self.loss_idt = torch.tensor([0.0]).cuda()

        loss_G =  lambda_G_teach * self.loss_teach + self.loss_adv_G +  lambda_G_Itv * self.G_Itv_loss \
                            + lambda_G_reblur * self.loss_reblur + lambda_G_global * self.loss_adv_globalG
   
        self.optimizer_G.zero_grad()
        loss_G.backward()
        if self.config['model']['g_name'] == 'MPRnet':
            torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), 0.01)
        self.optimizer_G.step()

    def get_loss(self):
        
        return OrderedDict([
                            ('D_sharp',self.loss_d_sharp.item()),
                            ('D_fake_sharp',self.loss_d_fake_sharp.item()),
                            ('D_blur',self.loss_d_blur.item()),
                            ('loss_D_reblur',self.loss_d_reblur.item()),
                            ('G_fS_S',self.loss_adv_G_fS.item()),
                            ('loss_global_G',self.loss_adv_globalG.item()),
                            ('loss_G_Itv',self.G_Itv_loss.item()),
                            # ('loss_adv_G',self.loss_adv_G.item()),
                            ('loss_G_reblur',self.loss_reblur.item()),
                            ('loss_teacher',self.loss_teach.item()),
                            ])
    
    def test(self, validation = False):
        with torch.no_grad():
            # self.fake_S = self.net_G(self.real_B)
            B,C,H,W = self.real_S.shape
            # self.mmap_real_B, offsets = self.net_M(self.real_B)
            # self.fake_B = self.blur_net(self.real_S, offsets)
            self.fake_S = self.net_G(self.real_B)
            
            mmap_real_B = self.net_M(self.real_B)
            mmap_real_S = self.net_M(self.real_S)
            if self.config['train']['relative_reblur']:
                mmap_B_S = self.vec_diff(mmap_real_B, mmap_real_S)
            else:
                mmap_B_S = mmap_real_B
            self.fake_B, _ = self.blur_net(self.real_S, mmap_B_S)

            mmap_fake_S = self.net_M(self.fake_S)
            print(torch.abs(mmap_fake_S).max(), torch.abs(mmap_real_B).max(), torch.abs(mmap_real_S).max())
            mmap_B_fS = self.vec_diff(mmap_real_B,mmap_fake_S)
            self.fake_B_from_fS, _ = self.blur_net(self.fake_S, mmap_real_B)
            # self.fake_B = fakeB_from_fakeS

            # import ipdb; ipdb.set_trace()
            # print("real B mean:",torch.mean(torch.abs(mmap_real_B)))
            # print("real S mean:",torch.mean(torch.abs(mmap_real_S)))
            # print("fS mean:",torch.mean(torch.abs(mmap_fake_S)))
            
        # calculate PSNR
        def PSNR(img1, img2):
            MSE = self.MSE(img1,img2)
            return 10 * np.log10(1 / MSE.item())

        if validation:
            reblur_S_psnr = 0
            reblur_fS_psnr = 0
            sharp_psnr = 0
            reblur_S_psnr += PSNR(self.real_B,self.fake_B) 
            sharp_psnr += PSNR(self.real_S,self.fake_S) 
            reblur_fS_psnr += PSNR(self.real_B,self.fake_B_from_fS) 
            return (reblur_S_psnr,reblur_fS_psnr,sharp_psnr)
            
    def test_multi_inf(self, validation = False):
        inference_num = self.config['test']['inference_num']
        self.optimizer_M = torch.optim.Adam( self.net_M.parameters(), lr=self.config['train']['lr_M'], betas=(0.9, 0.999) )
        self.optimizer_G = torch.optim.Adam( self.net_G.parameters(), lr=self.config['train']['lr_G'], betas=(0.9, 0.999) )
        self.net_M.train()
        self.net_G.train()
        best_psnr = 0
        for i in range(inference_num):
            self.fake_S = self.net_G(self.real_B)
            # with torch.no_grad(): 
            #     mmap_real_B = self.net_M(self.real_B)
            #     mmap_real_S = self.net_M(self.real_S)
            #     if self.config['train']['relative_reblur']:
            #         mmap_B_S = self.vec_diff(mmap_real_B, mmap_real_S)
            #     else:
            #         mmap_B_S = mmap_real_B
            #     self.fake_B, _ = self.blur_net(self.real_S, mmap_B_S)

            #     mmap_fake_S = self.net_M(self.fake_S)
            #     mmap_B_fS = self.vec_diff(mmap_real_B,mmap_fake_S)
            #     self.fake_B_from_fS, _ = self.blur_net(self.fake_S, mmap_real_B)
                    
            self.update_M()
            self.update_G()
            reblur_S_psnr = PSNR(self.real_B,self.fake_B) 
            sharp_psnr = PSNR(self.real_S,self.fake_S) 
            print('PSNR on step %d:'%i, sharp_psnr)
            # if sharp_psnr > best_psnr:
            #     best_psnr = sharp_psnr
            #     best_pic = self.fake_S
            reblur_fS_psnr = PSNR(self.real_B,self.fake_B_from_fS) 
        # self.fake_S = best_pic
        return (reblur_S_psnr,reblur_fS_psnr,sharp_psnr)

    def save(self,epoch):
        save_d_filename = 'D_net_%s.pth'%epoch
        save_naturalD_filename = 'naturalD_net_%s.pth'%epoch
        save_g_filename = 'G_net_%s.pth'%epoch
        if len(self.config['gpu'])>1:
            torch.save(self.net_M.module.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_d_filename))
            torch.save(self.net_naturalD.module.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_naturalD_filename))
            torch.save(self.net_G.module.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_g_filename))
        else:
            torch.save(self.net_M.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_d_filename))
            torch.save(self.net_naturalD.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_naturalD_filename))
            torch.save(self.net_G.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_g_filename))
        
        #     self.deblur_net.to(self.device)

    def load(self, config):
        load_path = os.path.join(config['checkpoints'], config['model_name'])
        load_D_file = load_path + '/' + 'D_net_%s.pth'%config['which_epoch']
        load_G_file = load_path + '/' + 'G_net_%s.pth'%config['which_epoch']
        load_naturalD_file = load_path + '/' + 'naturalD_net_%s.pth'%config['which_epoch']
        
        if len(self.config['gpu'])>1:
            if not config['load_only_G']:
                self.net_M.module.load_state_dict(torch.load(load_D_file))
                self.net_naturalD.module.load_state_dict(torch.load(load_naturalD_file))
                print('--------load model %s success!-------'%load_D_file)
                print('--------load model %s success!-------'%load_naturalD_file)
            self.net_G.module.load_state_dict(torch.load(load_G_file))
        else:
            if not config['load_only_G']:
                self.net_M.load_state_dict(torch.load(load_D_file))
                self.net_naturalD.load_state_dict(torch.load(load_naturalD_file))
                print('--------load model %s success!-------'%load_D_file)
                print('--------load model %s success!-------'%load_naturalD_file)
            self.net_G.load_state_dict(torch.load(load_G_file))
        
        print('--------load model %s success!-------'%load_G_file)


    def schedule_lr(self, epoch,tot_epoch):
        # scheduler
        # print("current learning rate:%.7f"%self.scheduler.get_lr())
        self.get_current_lr_from_epoch(self.optimizer_G, self.config['train']['lr_G'], epoch, tot_epoch)
        self.get_current_lr_from_epoch(self.optimizer_M, self.config['train']['lr_D'], epoch, tot_epoch)
        self.get_current_lr_from_epoch(self.optimizer_naturalD, self.config['train']['lr_D'], epoch, tot_epoch)

    def get_current_lr_from_epoch(self,optimizer, lr, epoch, tot_epoch):
        # current_lr = lr * (0.9**(epoch//decrease_step))
        current_lr = lr * (1 - epoch/tot_epoch)
        # if epoch > 500:
        #     current_lr = 0.000001
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        print("current learning rate:%.7f"%(current_lr))

    def get_current_visuals(self):
        real_B = utils.tensor2im(self.real_B)
        real_S = utils.tensor2im(self.real_S)
        fake_S = utils.tensor2im(self.fake_S)
        fake_B = utils.tensor2im(self.fake_B_from_fS)
        return OrderedDict([('real_B',real_B),('real_S',real_S),('fake_S',fake_S),('fake_B',fake_B)])
    


    def get_image_path(self):
        return {'B_path':self.B_path}

    def get_mmap(self, input):
        mmap = self.net_M(input)
        return mmap
