import torch
import torch.nn as nn
import numpy as np
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.functional as F
import functools
from .functions import ReverseLayerF
from .UNet_discriminator import UNetDiscriminator
from .DCN_v2.modules.modulated_deform_conv import ModulatedDeformConv_blur
from .MPRNet import MPRNet
from .MIMOUNet import MIMOUNet,MIMOUNetPlus



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='xavier', init_gain=1, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids)>1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net



def define_net_G(config):
    generator_name = config['model']['g_name']
    if generator_name == 'DMPHN':
        model_g = DMPHN_deblur()
    elif generator_name == 'MPRnet':
        model_g = MPRNet()
    elif generator_name == 'MIMO':
        model_g = MIMOUNetPlus()
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)
    model_g = init_net(model_g,gpu_ids=config['gpu'])
    return model_g

def define_net_M(config):
    if config['model']['norm'] != None:
        norm_layer = get_norm_layer(norm_type=config['model']['norm'])
    else:
        norm_layer = None
    discriminator_name = config['model']['d_name']
    if discriminator_name == 'unet':
        model_d = UNetDiscriminator(inChannels=3, outChannels=2,use_sigmoid=config['model']['use_sigmoid'])
    
    elif discriminator_name == 'Offset':
        model_d = OffsetNet(input_nc=3,nf=16,output_nc=2,offset_method=config['model']['offset_mode'])
    
    else:
        raise ValueError("discriminator Network [%s] not recognized." % discriminator_name)
    model_d = init_net(model_d,gpu_ids=config['gpu'])
    return model_d

def define_natural_D(config, input_nc=3, ndf=64, n_layers_D=3, norm='instance', use_sigmoid=False, num_D=2, getIntermFeat=False):        
    norm_layer = get_norm_layer(norm_type=norm)   
    patch_gan = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat=True)   
 
    
    netD = init_net(patch_gan, gpu_ids=config['gpu'])
    return netD


def define_blur(input_nc=3, output_nc=3, offset_num=15,gpu_ids=[]):
    net_blur = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
    net_blur = BlurNet(offsets_num=offset_num)

    net_blur = init_net(net_blur, gpu_ids=gpu_ids,initialize_weights=False)
    return net_blur

  
############   Classes     ##############
class BlurNet(nn.Module):
    def __init__(self,offsets_num=15):
        super(BlurNet,self).__init__()
        kernel_size = 1
        self.offsets_num = offsets_num
        self.Dcn = ModulatedDeformConv_blur(in_channels=1, out_channels=1, kernel_size=kernel_size,
						stride=1, padding=0, deformable_groups=1)
        
    def linear_traj(self,offset10,offset12):
        B,C,H,W = offset10.size()
        N = self.offsets_num//2
        t12 = torch.arange(1,N+1,step=1,dtype=torch.float32).cuda()
        t01 = torch.arange(N,0,step=-1,dtype=torch.float32).cuda()
        t12 = t12/N
        t01 = t01/N
        t12 = t12.view(-1,1,1,1)
        t01 = t01.view(-1,1,1,1)
        offset10 = offset10.view(B,1,2,H,W)
        offset12 = offset12.unsqueeze(1)
        offset_12_traj = t12 * offset12
        offset_01_traj = t01 * offset10
        offset_12_traj = offset_12_traj.view(B,-1,H,W)
        offset_01_traj = offset_01_traj.view(B,-1,H,W)
        return offset_01_traj,offset_12_traj
    
    def Quadra_traj(self,offset10,offset12):
        B,C,H,W = offset10.size()
        N = self.offsets_num//2
        t = torch.arange(1,N+1,step=1,dtype=torch.float32).cuda()
        t = t/N
        t = t.view(-1,1,1,1)
        offset10 = offset10.view(B,1,2,H,W)
        offset12 = offset12.unsqueeze(1)
        offset_12N = 0.5 * ((t + t**2)*offset12 - (t - t**2)*offset10)
        offset_10N = 0.5 * ((t + t**2)*offset10 - (t - t**2)*offset12)
        offset_12N = offset_12N.view(B,-1,H,W)
        offset_10N = offset_10N.view(B,-1,H,W)

        return offset_10N,offset_12N

    def forward(self,fake_S,blurmap):
        B,C,H,W = blurmap.shape
        
        # blurmap to offsets
        if C == 2:
            offset_SPoint = blurmap * 20 
            offset_EPoint = 0 - offset_SPoint
            offset_S_0, offset_0_E = self.linear_traj(offset_SPoint,offset_EPoint)
            zeros = torch.zeros(B,2,H,W).cuda()
        elif C == 4:
            blurmap = blurmap * 10
            offset_SPoint = blurmap[:,:2,:,:]
            offset_EPoint = blurmap[:,2:,:,:]
            offset_S_0, offset_0_E = self.Quadra_traj(offset_SPoint,offset_EPoint)
            zeros = torch.zeros(B,2,H,W).cuda()
        offsets = torch.cat((offset_S_0,zeros,offset_0_E),dim=1)
        # warping S to blur_offset_i
        offset_N = torch.chunk(offsets, self.offsets_num,dim=1)
        fake_B_n = torch.zeros(B,3*self.offsets_num,H,W).cuda()
        for i,offset_i in enumerate(offset_N):
            o1, o2 = torch.chunk(offset_i, 2, dim=1) 
            offset_i = torch.cat((o1, o2), dim=1)
            mask = torch.ones(B,1,H,W).cuda() 
            fake_B_n[:,i*3:(i+1)*3,:,:] = self.Dcn(fake_S,offset_i,mask)
        
        # fake_B = fake_B_n/self.offsets_num
        fake_B_n = fake_B_n.view(B,self.offsets_num,-1,H,W)
        fake_B = torch.sum(fake_B_n,dim=1)/self.offsets_num
        
        return fake_B, offsets


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, use_bias, norm_layer):
        super(ResnetBlock, self).__init__()

        padAndConv_1 = [
                nn.ReplicationPad2d(2),
                nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)]

        padAndConv_2 = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)]

        if norm_layer:
            blocks = padAndConv_1 + [
                norm_layer(dim),
                nn.ReLU(True)
            ]  + padAndConv_2 + [
                norm_layer(dim)]
        else:
            blocks = padAndConv_1 + [
                nn.ReLU(True)
            ]  + padAndConv_2 
        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        out = x + self.conv_block(x)
        return out

def TriResblock(input_nc, norm_layer=None, use_bias=True):
    Res1 =  ResnetBlock(input_nc, padding_type='reflect', use_bias=use_bias, norm_layer=norm_layer)
    Res2 =  ResnetBlock(input_nc, padding_type='reflect', use_bias=use_bias, norm_layer=norm_layer)
    Res3 =  ResnetBlock(input_nc, padding_type='reflect', use_bias=use_bias, norm_layer=norm_layer)
    return nn.Sequential(Res1,Res2,Res3)

def conv_TriResblock(input_nc,out_nc,stride, use_bias=True, norm_layer=None):
    Relu = nn.ReLU(True)
    if stride==1:
        pad = nn.ReflectionPad2d(2)
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=1,padding=0,bias=use_bias)
    elif stride==2:
        pad = nn.ReflectionPad2d((1,2,1,2))
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=2,padding=0,bias=use_bias)
    tri_resblock = TriResblock(out_nc, norm_layer=norm_layer)
    return nn.Sequential(pad,conv,Relu,tri_resblock)
        

class OffsetNet(nn.Module):
    # offset for Start and End Points, then calculate a quadratic function
    def __init__(self, input_nc, nf, output_nc, norm_layer=None,offset_method='lin'):
        super(OffsetNet,self).__init__()
        self.input_nc = input_nc
        self.nf = nf
        self.offset_method = offset_method
        if offset_method == 'quad' or offset_method == 'bilin':
            output_nc = 2 * 2
        elif offset_method == 'lin':
            output_nc = 1 * 2
        
        use_bias = True
        self.pad_1 = nn.ReflectionPad2d((1,2,1,2))
        self.todepth = SpaceToDepth(block_size=2)
        self.conv_1 = conv_TriResblock(input_nc*4,nf,stride=1,use_bias=True,norm_layer=norm_layer)
        self.conv_2 = conv_TriResblock(nf,nf*2,stride=2,use_bias=True,norm_layer=norm_layer)
        self.conv_3 = conv_TriResblock(nf*2,nf*4,stride=2,use_bias=True,norm_layer=norm_layer)
        # self.conv_4 = conv_TriResblock(nf*4,nf*8,stride=1,use_bias=True)

        self.bottleneck_1 = Bottleneck(nf*4)
        self.uconv_1 = nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, 
                                        bias=use_bias)
        
        self.bottleneck_2 = Bottleneck(nf*4)        
        self.uconv_2 = nn.ConvTranspose2d(nf*4, nf, kernel_size=4, stride=2, padding=1, 
                                        bias=use_bias)
        self.bottleneck_3 = Bottleneck(nf*2)
        self.uconv_3 = nn.ConvTranspose2d(nf*2, nf*2, kernel_size=4, stride=2, padding=1, 
                                        bias=use_bias)
        self.conv_out_0 = nn.Conv2d(nf*2,output_nc,kernel_size=5,stride=1,padding=2,bias=use_bias)

    def forward(self,input):     
        scale_0 = input
        B,N,H,W = input.size()
        scale_0_depth = self.todepth(scale_0)
        d_conv1 = self.conv_1(scale_0_depth)
        d_conv2 = self.conv_2(d_conv1)
        d_conv3 = self.conv_3(d_conv2)

        d_conv3 = self.bottleneck_1(d_conv3)
        u_conv1 = self.uconv_1(d_conv3)
        u_conv1 = F.leaky_relu(u_conv1,0.2,True) 
        u_conv1 = torch.cat((u_conv1 , d_conv2),dim=1)
        
        u_conv1 = self.bottleneck_2(u_conv1)
        u_conv2 = self.uconv_2(u_conv1)
        u_conv2 = F.leaky_relu(u_conv2,0.2,True)
        u_conv2 = torch.cat((u_conv2 , d_conv1),dim=1)

        u_conv2 = self.bottleneck_3(u_conv2)
        u_conv3 = self.uconv_3(u_conv2)
        out = self.conv_out_0(F.relu(u_conv3))

        # if self.offset_method == 'quad':
        #     out = torch.cat([torch.abs(out[:,0:2]),-torch.abs(out[:,2:])], dim=1)
        #     out = torch.cat([out[:,:2],-out[:,2:]],dim=1)
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=2, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)   










class Bottleneck(nn.Module):
    def __init__(self,nChannels,kernel_size=3):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(nChannels, nChannels*2, kernel_size=1, 
                                padding=0, bias=True)
        self.lReLU1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(nChannels*2, nChannels, kernel_size=kernel_size, 
                                padding=(kernel_size-1)//2, bias=True)
        self.lReLU2 = nn.LeakyReLU(0.2, True)
        self.model = nn.Sequential(self.conv1,self.lReLU1,self.conv2,self.lReLU2)
    def forward(self,x):
        out = self.model(x)
        return out

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self,x):        
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x

class DMPHN_deblur(nn.Module):
    def __init__(self):
        super(DMPHN_deblur,self).__init__()
        
        self.encoder_lv1 = Encoder()
        self.encoder_lv2 = Encoder()
        self.encoder_lv3 = Encoder()
        self.encoder_lv4 = Encoder()

        self.decoder_lv1 = Decoder()
        self.decoder_lv2 = Decoder()
        self.decoder_lv3 = Decoder()
        self.decoder_lv4 = Decoder()

    def forward(self, image):
        images_lv1 = image
        H = images_lv1.size(2)
        W = images_lv1.size(3)

        images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]
        images_lv4_1 = images_lv3_1[:,:,0:int(H/4),:]
        images_lv4_2 = images_lv3_1[:,:,int(H/4):int(H/2),:]
        images_lv4_3 = images_lv3_2[:,:,0:int(H/4),:]
        images_lv4_4 = images_lv3_2[:,:,int(H/4):int(H/2),:]
        images_lv4_5 = images_lv3_3[:,:,0:int(H/4),:]
        images_lv4_6 = images_lv3_3[:,:,int(H/4):int(H/2),:]
        images_lv4_7 = images_lv3_4[:,:,0:int(H/4),:]
        images_lv4_8 = images_lv3_4[:,:,int(H/4):int(H/2),:]

        feature_lv4_1 = self.encoder_lv4(images_lv4_1)

        feature_lv4_2 = self.encoder_lv4(images_lv4_2)
        feature_lv4_3 = self.encoder_lv4(images_lv4_3)
        feature_lv4_4 = self.encoder_lv4(images_lv4_4)
        feature_lv4_5 = self.encoder_lv4(images_lv4_5)
        feature_lv4_6 = self.encoder_lv4(images_lv4_6)
        feature_lv4_7 = self.encoder_lv4(images_lv4_7)
        feature_lv4_8 = self.encoder_lv4(images_lv4_8)
        feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
        feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
        feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
        feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)
        feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
        feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
        feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
        residual_lv4_top_left = self.decoder_lv4(feature_lv4_top_left)
        residual_lv4_top_right = self.decoder_lv4(feature_lv4_top_right)
        residual_lv4_bot_left = self.decoder_lv4(feature_lv4_bot_left)
        residual_lv4_bot_right = self.decoder_lv4(feature_lv4_bot_right)

        feature_lv3_1 = self.encoder_lv3(images_lv3_1 + residual_lv4_top_left)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2 + residual_lv4_top_right)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3 + residual_lv4_bot_left)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4 + residual_lv4_bot_right)
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3) + feature_lv4_top
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3) + feature_lv4_bot
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        residual_lv3_top = self.decoder_lv3(feature_lv3_top)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot)

        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
        residual_lv2 = self.decoder_lv2(feature_lv2)

        feature_lv1 = self.encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
        deblur_image = self.decoder_lv1(feature_lv1)
        return deblur_image