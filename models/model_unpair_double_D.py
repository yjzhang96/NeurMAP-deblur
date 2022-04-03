import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import os
import torch.nn.functional as F
from . import networks_new
from collections import OrderedDict
from utils import utils_new as utils
from .losses import get_loss, SSIMLoss
from .schedulers import WarmRestart,LinearDecay
from utils.image_pool import ImagePool


networks = networks_new

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
        self.net_D = networks.define_net_D(config)
        self.net_globalD = networks.define_global_D(config)
        # self.net_bmap_pretrain = networks.define_net_offset(config)
        self.blur_net = networks.define_blur(input_nc=3, output_nc=3, n_offset=self.n_offset, gpu_ids=config['gpu'])      # deformable
        # self.blur_net_layer = networks.define_blur_layer(input_nc=3, output_nc=3, n_offset=self.n_offset, gpu_ids=config['gpu'])      # deformable
                
        ###Loss and Optimizer
        self.criterion_adv = get_loss('blur-gan')
        self.criterion_reblur = get_loss('reblur')
        
        # global D lsGAN/RaGAN-ls
        self.criterion_lsgan = get_loss('lsgan')
        # self.criterion_lsgan = get_loss('ragan-ls')

        self.criterion_content = get_loss('l1')
        self.MSE = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.SSIMloss = SSIMLoss('MSSSIM')
        

        ### Image pool
        self.real_pool = ImagePool(50)

        # load_bmap_file = config['bmap_model']
        # load_G_model = config['G_model']
        # if len(config['gpu']) > 1:
        #     self.net_G.module.load_state_dict(torch.load(load_G_model))
        # else:
        #     self.net_G.load_state_dict(torch.load(load_G_model))
        # print('--------load model %s success!-------'%load_G_model)

        if config['is_training']:
            self.optimizer_D = torch.optim.Adam( self.net_D.parameters(), lr=config['train']['lr_D'], betas=(0.9, 0.999) )
            self.optimizer_G = torch.optim.Adam( self.net_G.parameters(), lr=config['train']['lr_G'], betas=(0.9, 0.999) )
            self.optimizer_globalD = torch.optim.Adam( self.net_globalD.parameters(), lr=config['train']['lr_D'], betas=(0.9, 0.999) )
            
        if config['resume_train']:
            print("------loading learning rate------")
    
            self.get_current_lr_from_epoch(self.optimizer_G, config['train']['lr_G'], config['start_epoch'], config['epoch'])
            self.get_current_lr_from_epoch(self.optimizer_D, config['train']['lr_D'], config['start_epoch'], config['epoch'])
            self.get_current_lr_from_epoch(self.optimizer_globalD, config['train']['lr_D'], config['start_epoch'], config['epoch'])


    def set_input(self,batch_data):
        self.real_B = batch_data['B'].to(self.device)
        if batch_data['gt'][0]:
            self.real_S_exist = True
            self.real_S = batch_data['S'].to(self.device)
        else:
            self.real_S_exist = False
            self.real_S = batch_data['S'].to(self.device)
        self.B_path = batch_data['B_path']


    def optimize(self):
        self.fake_S = self.net_G(self.real_B)
        self.update_D()
        if self.config['train']['global_D']:
            self.update_globalD()
        else:
            self.loss_global_D = torch.tensor(0.0)
        self.update_G()


    def optimize_D(self):
        # forward
        self.fake_S = self.net_G(self.real_B)
        self.update_D()
        self.update_globalD()
    
    def optimize_G(self):
        # forward
        self.fake_S = self.net_G(self.real_B)
        self.update_G()

    def warmup_optimize_G(self, train_D=False):
        self.fake_S = self.net_G(self.real_B)
        self.update_G(warmup=True)

    @staticmethod
    def vec_norm(vec):
        B,C,H,W = vec.shape
        assert C == 2
        # vec_trans = torch.transpose(vec,(0,2,3,1))
        x,y = torch.chunk(vec,2,dim=1)
        
        x_norm = torch.where(y<0,-x,x)
        y_norm = torch.where(y<0,-y,y)
        vec_norm = torch.cat((x_norm,y_norm),dim=1)
        return vec_norm

    
    def vec_diff(self, vec1, vec2):
        vec1_abs = self.vec_norm(vec1)
        vec2_abs = self.vec_norm(vec2)
        diff = vec1_abs - vec2_abs
        return diff

    def update_D(self,warmup=False):
        # import ipdb; ipdb.set_trace()
        B,C,H,W = self.real_B.shape
        lambda_d_S = self.config['train']['lambda_D_sharp']
        lambda_d_fS = self.config['train']['lambda_D_f_sharp']
        lambda_d_B = self.config['train']['lambda_D_blur']
        lambda_d_tv = self.config['train']['lambda_D_tv']

        bmap_real_B = self.net_D(self.real_B)
        # self.bmap_real_B_gt = self.net_bmap_pretrain(self.real_B)
        bmap_real_S = self.net_D(self.real_S)
        fake_S_detach = self.fake_S.detach()
        bmap_fake_S = self.net_D(fake_S_detach)
        
        bmap_real_B_abs = self.vec_norm(bmap_real_B)
        # bmap_real_B_gt_abs = self.vec_norm(self.bmap_real_B_gt)
        bmap_real_S_abs = self.vec_norm(bmap_real_S)
        bmap_fake_S_abs = self.vec_norm(bmap_fake_S)
        self.bmaps = {'real_B':bmap_real_B_abs,'fake_S':bmap_fake_S_abs,'real_S':bmap_real_S_abs}
        # self.loss_d_sharp, self.loss_d_fake_sharp = self.criterion_adv.get_d_loss(bmap_real_S_abs,bmap_real_B_gt_abs,bmap_fake_S_abs)

        one_tensor = torch.ones_like(bmap_real_B).cuda()
        zero_tensor = torch.zeros_like(bmap_real_B).cuda()
        # add image pool
        self.real_pool.add(bmap_real_S_abs)
        ## offset to GT
        # self.loss_d_sharp = self.L1loss(bmap_real_S_abs, zero_tensor)
        # self.loss_d_fake_sharp = self.L1loss(bmap_fake_S_abs, bmap_real_B_gt_abs)
        # self.loss_d_blur = self.L1loss(bmap_real_B_abs, bmap_real_B_gt_abs)
        # self.loss_adv_D  = lambda_d_S * self.loss_d_sharp \
        #                         + lambda_d_fS * self.loss_d_fake_sharp + lambda_d_B * self.loss_d_blur
        
        ## offset to 0/1
        self.loss_d_sharp = self.L1loss(bmap_real_S_abs, zero_tensor)
        bmap_real_B_abs_detach = bmap_real_B_abs.detach()
        self.loss_d_fake_sharp = self.L1loss(bmap_fake_S_abs, bmap_real_B_abs_detach)
        self.loss_d_blur = self.L1loss(torch.abs(bmap_real_B_abs), one_tensor)
        self.loss_adv_D  = lambda_d_S * self.loss_d_sharp \
                                + lambda_d_fS * self.loss_d_fake_sharp + lambda_d_B * self.loss_d_blur
        
        # loss Map_fS is smaller than Map_B
        Mag_fake_S = torch.sqrt(bmap_fake_S_abs[:,0,:,:]**2 + bmap_fake_S_abs[:,1,:,:]**2)
        Mag_real_B = torch.sqrt(bmap_real_B_abs[:,0,:,:]**2 + bmap_real_B_abs[:,1,:,:]**2)
        Mag_diff = Mag_fake_S - Mag_real_B
        Mag_gt = torch.where(Mag_diff>0, Mag_diff, torch.zeros_like(Mag_diff))
        self.loss_Mag_gt = torch.mean(Mag_gt)        
        
        
        # reblur loss       
        # import ipdb; ipdb.set_trace()
        if self.config['train']['relative_reblur']:
            bmap_rB_fS = bmap_real_B_abs - bmap_fake_S_abs
            self.fake_B_from_fS, offsets = self.blur_net(fake_S_detach, bmap_rB_fS)
            # bmap_rB_rS = bmap_real_B_abs - bmap_real_S_abs.detach()
            # self.fake_B_from_S , _ = self.blur_net(self.real_S[:B//2], bmap_rB_rS[:B//2])
        elif self.config['train']['absolute_reblur']:
            self.fake_B, offsets = self.blur_net(self.real_S, bmap_real_B_abs)
        

        B,C,H,W = self.real_B.shape
        offsets	= offsets.view(B,15,-1,H,W)
        
        # # spatial tv_loss across one offset
        # lambda_tv = 0.002
        self.tv_loss = self.L1loss(offsets[:,:,:,:,:-1],offsets[:,:,:,:,1:]) + \
                        self.L1loss(offsets[:,:,:,:-1,:],offsets[:,:,:,1:,:])

        # # regulationl loss
        lambda_reg = 0.0002
        # lambda_reg = 0.0
        self.reg_loss = torch.mean(offsets[:,:,:,:,:]**2)
        
        # MSE loloss
        loss_MSE_fS = self.MSE(self.fake_B_from_fS,self.real_B)
        # loss_MSE_S = self.MSE(self.fake_B_from_S,self.real_B[:B//2])
        self.loss_MSE = loss_MSE_fS 

        # SSIM loss
        lambda_SSIM = 0.1
        ssim_loss_fS = 1 - self.SSIMloss.get_loss(self.fake_B_from_fS,self.real_B)
        # ssim_loss_S = 1 - self.SSIMloss.get_loss(self.fake_B_from_S,self.real_B[:B//2])
        self.ssim_loss = ssim_loss_fS 

        loss_reblur = lambda_SSIM * self.ssim_loss + self.loss_MSE \
                            + lambda_reg * self.reg_loss + lambda_d_tv * self.tv_loss 

        if self.config['train']['R1_grad_penalty']:
            R1_gamma = self.config['train']['lambda_R1_gamma']
            
            
            grad_real = autograd.grad(outputs=bmap_real_B.sum(), inputs=self.real_B, 
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            self.loss_R1_GP = 0.5 * R1_gamma * grad_penalty
            self.loss_total = self.loss_adv_D + self.loss_R1_GP
        else:
            self.loss_R1_GP = torch.tensor([0.0]) 
            self.loss_total = self.loss_adv_D   + loss_reblur + self.loss_Mag_gt
        # import ipdb; ipdb.set_trace()
        self.optimizer_D.zero_grad()
        self.loss_total.backward()

        self.optimizer_D.step()
        
    def update_globalD(self):
        B,C,H,W = self.real_B.shape

        pred_fake = self.net_globalD(self.fake_S.detach())
        loss_D_fake = self.criterion_lsgan(pred_fake, False)

        # real_batch = torch.cat((self.real_S[0:B:2],self.real_B[1:B:2]),dim=0)
        # import ipdb; ipdb.set_trace()
        pred_real = self.net_globalD(self.real_S)
        loss_D_real = self.criterion_lsgan(pred_real, True)
        self.loss_global_D = loss_D_fake + loss_D_real 
        # self.loss_global_D = self.criterion_lsgan(self.net_globalD,self.fake_S,self.real_S)
        self.optimizer_globalD.zero_grad()
        self.loss_global_D.backward()
        self.optimizer_globalD.step()
    
    
    def update_G(self,warmup=False):
        B,C,H,W = self.real_B.shape

        lambda_G_fS = self.config['train']['lambda_G_f_sharp']
        lambda_G_idt = self.config['train']['lambda_G_idt']
        lambda_G_tv = self.config['train']['lambda_G_tv']
        lambda_G_content = self.config['train']['lambda_G_content']
        lambda_G_reblur = self.config['train']['lambda_G_reblur']
        lambda_G_global = self.config['train']['lambda_G_global']

        bmap_fake_S = self.net_D(self.fake_S)
        bmap_real_B = self.net_D(self.real_B)
        bmap_fake_S_abs = self.vec_norm(bmap_fake_S)
        zero_tensor = torch.zeros_like(bmap_real_B).cuda()
        self.loss_adv_G_fS =  self.L1loss(torch.abs(bmap_fake_S_abs), zero_tensor)
        ## G tv loss
        self.G_tv_loss = self.L1loss(bmap_fake_S_abs[:,:,:,:-1],bmap_fake_S_abs[:,:,:,1:]) + \
                        self.L1loss(bmap_fake_S_abs[:,:,:-1,:],bmap_fake_S_abs[:,:,1:,:])

        ## relativistic loss
        # real_pool_query = torch.abs(self.real_pool.query())
        # real_pool_mean = torch.mean(real_pool_query)
        # # real_pool_mean = real_pool_mean.view(1,-1,1,1)
        # # import ipdb; ipdb.set_trace()
        # self.loss_adv_G_fS =  torch.mean((torch.abs(bmap_fake_S_abs) - real_pool_mean)**2)
        
        self.loss_adv_G = lambda_G_fS * self.loss_adv_G_fS + lambda_G_tv * self.G_tv_loss
        # import ipdb; ipdb.set_trace()

        # global G loss
        if self.config['train']['global_D']:
            pred_fake = self.net_globalD(self.fake_S)
            self.loss_adv_globalG = self.criterion_lsgan(pred_fake, True)
            # self.loss_adv_globalG = self.criterion_lsgan.get_g_loss(self.net_globalD,self.fake_S,self.real_S)
        else:
            self.loss_adv_globalG = torch.tensor(0.0)

        # reblur loss to train G?
        if self.config['train']['relative_reblur']:
            # import ipdb; ipdb.set_trace()
            bmap_real_B_abs = self.vec_norm(bmap_real_B)
            reblur_bmap = bmap_real_B_abs - bmap_fake_S_abs
        elif self.config['train']['absolute_reblur']:
            
            reblur_bmap = bmap_real_B
        reblur_bmap_detach = reblur_bmap.detach()
        self.fake_B, _ = self.blur_net(self.fake_S, reblur_bmap_detach)
        self.loss_reblur = self.criterion_reblur(self.real_B, self.fake_B)

        # apply content loss to warm start net_G
        # self.loss_content = self.L1loss(self.real_S, self.fake_S)            
        if self.config['train']['identical_loss']:
            real_S_idt = self.net_G(self.real_S)
            self.loss_idt = self.L1loss(self.real_S, real_S_idt)            
        else:
            self.loss_idt = torch.tensor([0.0]).cuda()

        loss_G =  self.loss_adv_G + \
                            + lambda_G_reblur * self.loss_reblur + lambda_G_global * self.loss_adv_globalG
   
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

    def get_loss(self):
        
        return OrderedDict([
                            ('D_sharp',self.loss_d_sharp.item()),
                            ('D_fake_sharp',self.loss_d_fake_sharp.item()),
                            ('D_blur',self.loss_d_blur.item()),
                            ('loss_Mag_gt',self.loss_Mag_gt.item()),
                            ('G_fS_S',self.loss_adv_G_fS.item()),
                            ('G_tv_loss',self.G_tv_loss.item()),
                            ('loss_global_G',self.loss_adv_globalG.item()),
                            ('loss_idt',self.loss_idt.item()),
                            # ('loss_adv_G',self.loss_adv_G.item()),
                            ('loss_reblur',self.loss_reblur.item()),
                            ])
    
    def test(self, validation = False):
        with torch.no_grad():
            # self.fake_S = self.net_G(self.real_B)
            B,C,H,W = self.real_S.shape
            # self.bmap_real_B, offsets = self.net_D(self.real_B)
            # self.fake_B = self.blur_net(self.real_S, offsets)
            self.fake_S = self.net_G(self.real_B)
            
            bmap_real_B = self.net_D(self.real_B)
            bmap_real_S = self.net_D(self.real_S)
            if self.config['train']['relative_reblur']:
                bmap_B_S = self.vec_diff(bmap_real_B, bmap_real_S)
            else:
                bmap_B_S = bmap_real_B
            self.fake_B, _ = self.blur_net(self.real_S, bmap_B_S)

            bmap_fake_S = self.net_D(self.fake_S)
            bmap_B_fS = self.vec_diff(bmap_real_B,bmap_fake_S)
            self.fake_B_from_fS, _ = self.blur_net(self.fake_S, bmap_B_fS)
            # self.fake_B = fakeB_from_fakeS

            # import ipdb; ipdb.set_trace()
            # print("real B mean:",torch.mean(torch.abs(bmap_real_B)))
            # print("real S mean:",torch.mean(torch.abs(bmap_real_S)))
            # print("fS mean:",torch.mean(torch.abs(bmap_fake_S)))
            
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

    def save(self,epoch):
        save_d_filename = 'D_net_%s.pth'%epoch
        save_globald_filename = 'globalD_net_%s.pth'%epoch
        save_g_filename = 'G_net_%s.pth'%epoch
        if len(self.config['gpu'])>1:
            torch.save(self.net_D.module.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_d_filename))
            torch.save(self.net_globalD.module.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_globald_filename))
            torch.save(self.net_G.module.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_g_filename))
        else:
            torch.save(self.net_D.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_d_filename))
            torch.save(self.net_globalD.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_globald_filename))
            torch.save(self.net_G.state_dict(),os.path.join(self.config['checkpoints'], self.config['model_name'],save_g_filename))
        
        #     self.deblur_net.to(self.device)

    def load(self, config):
        load_path = os.path.join(config['checkpoints'], config['model_name'])
        load_D_file = load_path + '/' + 'D_net_%s.pth'%config['which_epoch']
        load_G_file = load_path + '/' + 'G_net_%s.pth'%config['which_epoch']
        load_globalD_file = load_path + '/' + 'globalD_net_%s.pth'%config['which_epoch']
        
        if len(self.config['gpu'])>1:
            if not config['load_only_G']:
                self.net_D.module.load_state_dict(torch.load(load_D_file))
                self.net_globalD.module.load_state_dict(torch.load(load_globalD_file))
                print('--------load model %s success!-------'%load_D_file)
                print('--------load model %s success!-------'%load_globalD_file)
            self.net_G.module.load_state_dict(torch.load(load_G_file))
        else:
            if not config['load_only_G']:
                self.net_D.load_state_dict(torch.load(load_D_file))
                self.net_globalD.load_state_dict(torch.load(load_globalD_file))
                print('--------load model %s success!-------'%load_D_file)
                print('--------load model %s success!-------'%load_globalD_file)
            self.net_G.load_state_dict(torch.load(load_G_file))
        
        print('--------load model %s success!-------'%load_G_file)


    def schedule_lr(self, epoch,tot_epoch):
        # scheduler
        # print("current learning rate:%.7f"%self.scheduler.get_lr())
        self.get_current_lr_from_epoch(self.optimizer_G, self.config['train']['lr_G'], epoch, tot_epoch)
        self.get_current_lr_from_epoch(self.optimizer_D, self.config['train']['lr_D'], epoch, tot_epoch)
        self.get_current_lr_from_epoch(self.optimizer_globalD, self.config['train']['lr_D'], epoch, tot_epoch)

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
    
    def get_tensorboard_images(self):
        print("bmap_B mean:",torch.abs(self.bmaps['real_B'][-1]).mean())
        print("bmap_S mean:",torch.abs(self.bmaps['real_S'][-1]).mean())
        print("bmap_fS mean:",torch.abs(self.bmaps['fake_S'][-1]).mean())
        bmap_B_S = self.bmaps['real_B'][-1] - self.bmaps['real_S'][-1]
        bmap_B_fS = self.bmaps['real_B'][-1] - self.bmaps['fake_S'][-1]
        bmap_S = self.bmaps['real_S'][-1]

        bmap_realB_vis = torch.tensor(utils.bmap2heat(bmap_B_S)/255)
        bmap_fakeS_vis = torch.tensor(utils.bmap2heat(bmap_B_fS)/255)
        bmap_realS_vis = torch.tensor(utils.bmap2heat(bmap_S)/255)
        bmap_realB_vis = bmap_realB_vis.permute(2,0,1)
        bmap_fakeS_vis = bmap_fakeS_vis.permute(2,0,1)
        bmap_realS_vis = bmap_realS_vis.permute(2,0,1)
        # import ipdb; ipdb.set_trace()

        real_B = torch.clamp(self.real_B,min=0,max=1)
        real_S = torch.clamp(self.real_S,min=0,max=1)
        fake_S = torch.clamp(self.fake_S,min=0,max=1)
        fake_B_from_fS = torch.clamp(self.fake_B_from_fS,min=0,max=1)
        bmap_vis = OrderedDict([('real_B',bmap_realB_vis),('real_S',bmap_realS_vis),('fake_S',bmap_fakeS_vis)])
        unpaired_train_img = OrderedDict([('real_B',real_B[-1]),('real_S',real_S[-1]),('fake_S',fake_S[-1]),('fake_B',fake_B_from_fS[-1])])
        return unpaired_train_img, bmap_vis

    def get_image_path(self):
        return {'B_path':self.B_path}

    def get_bmap(self, input):
        bmap = self.net_D(input)
        return self.vec_norm(bmap)
        # return bmap
