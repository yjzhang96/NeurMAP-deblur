import torch
import torch.autograd as autograd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

# from util.image_pool import ImagePool
from pytorch_msssim import msssim, ssim
from utils.image_pool import ImagePool
###############################################################################
# Functions
###############################################################################

class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)

class SSIMLoss():
	def __init__(self,metric):
		if metric == 'MSSSIM':
			self.criterion = msssim
		elif metric == 'SSIM':
			self.criterion = ssim
	
	def get_loss(self, fakeIm, realIm):
		return self.criterion(fakeIm,realIm)
        
class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss
            self.contentFunc = self.contentFunc()
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_loss(self, fakeIm, realIm):
        fakeIm = (fakeIm + 1) / 2.0
        realIm = (realIm + 1) / 2.0
        fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
        realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)


# class GANLoss(nn.Module):
#     def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
#                  tensor=torch.FloatTensor):
#         super(GANLoss, self).__init__()
#         self.real_label = target_real_label
#         self.fake_label = target_fake_label
#         self.real_label_var = None
#         self.fake_label_var = None
#         self.Tensor = tensor
#         if use_l1:
#             self.loss = nn.L1Loss()
#         else:
#             self.loss = nn.BCEWithLogitsLoss()

#     def get_target_tensor(self, input, target_is_real):
#         if target_is_real:
#             create_label = ((self.real_label_var is None) or
#                             (self.real_label_var.numel() != input.numel()))
#             if create_label:
#                 real_tensor = self.Tensor(input.size()).fill_(self.real_label)
#                 self.real_label_var = Variable(real_tensor, requires_grad=False)
#             target_tensor = self.real_label_var
#         else:
#             create_label = ((self.fake_label_var is None) or
#                             (self.fake_label_var.numel() != input.numel()))
#             if create_label:
#                 fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
#                 self.fake_label_var = Variable(fake_tensor, requires_grad=False)
#             target_tensor = self.fake_label_var
#         return target_tensor.cuda()

#     def __call__(self, input, target_is_real):
#         target_tensor = self.get_target_tensor(input, target_is_real)
#         return self.loss(input, target_tensor)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class DiscLoss(nn.Module):
    def name(self):
        return 'DiscLoss'

    def __init__(self):
        super(DiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_AB_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net.forward(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLoss(nn.Module):
    def name(self):
        return 'RelativisticDiscLoss'

    def __init__(self):
        super(RelativisticDiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.pred_fake = net.forward(fakeB)

        # Real
        self.pred_real = net.forward(realB)
        errG = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 0) +
                self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 1)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 1) +
                       self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 0)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLossLS(nn.Module):
    def name(self):
        return 'RelativisticDiscLossLS'

    def __init__(self):
        super(RelativisticDiscLossLS, self).__init__()

        self.criterionGAN = GANLoss(use_l1=True)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.pred_fake = net.forward(fakeB)

        # Real
        self.pred_real = net.forward(realB)
        errG = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) + 1) ** 2) +
                torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) - 1) ** 2)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) - 1) ** 2) +
                       torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) + 1) ** 2)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class DiscLossLS(DiscLoss):
    def name(self):
        return 'DiscLossLS'

    def __init__(self):
        super(DiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)

    def get_g_loss(self, net, fakeB, realB):
        return DiscLoss.get_g_loss(self, net, fakeB)

    def get_loss(self, net, fakeB, realB):
        return DiscLoss.get_loss(self, net, fakeB, realB)


class DiscLossWGANGP(DiscLossLS):
    def name(self):
        return 'DiscLossWGAN-GP'

    def __init__(self):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = 10

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty

class BlurmapDiscLoss(nn.Module):
    def name(self):
        return 'BlurmapDiscLoss'
    
    def __init__(self):
        super(BlurmapDiscLoss,self).__init__()
        self.criterion = nn.L1Loss()
        self.sharp_label = 0.0
        self.blur_label = 1.0
        self.sharp_label_map = None
        self.blur_label_map = None
        self.Tensor = torch.FloatTensor 
        self.lambda_fake_S = 1.0
        self.lambda_real_B = 0.0
        self.lambda_real_S = 0.0

    def get_g_loss(self,bmap_fake_S):
        create_label = ((self.sharp_label_map is None) or
                            (self.sharp_label_map.numel() != bmap_fake_S.numel()))
        if create_label:
            sharp_tensor = self.Tensor(bmap_fake_S.size()).fill_(self.sharp_label)
            self.sharp_label_map = Variable(sharp_tensor, requires_grad=False).cuda()
        
        loss_g = self.criterion(bmap_fake_S,self.sharp_label_map)
        # loss_g = torch.mean(torch.sqrt(bmap_fake_S[:,0,:,:] **2 + bmap_fake_S[:,1,:,:] **2))

        return loss_g

    def get_d_loss(self,bmap_real_S,bmap_real_B,bmap_fake_S,warmup=0):
        create_label = ((self.sharp_label_map is None) or (self.blur_label_map is None) or
                            (self.sharp_label_map.numel() != bmap_real_S.numel()))
        if create_label:
            sharp_tensor = self.Tensor(bmap_real_S.size()).fill_(self.sharp_label)
            self.sharp_label_map = Variable(sharp_tensor, requires_grad=False).cuda()
            blur_tensor = self.Tensor(bmap_real_S.size()).fill_(self.blur_label)
            self.blur_label_map = Variable(blur_tensor, requires_grad=False).cuda()           
        loss_d_sharp = self.criterion(bmap_real_S, self.sharp_label_map)
        loss_d_fake_sharp = self.criterion(bmap_fake_S, bmap_real_B)
        # loss_d_fake_sharp = self.criterion(bmap_fake_S, self.blur_label_map)
        
        # # blur map loss to be add?
        # loss_d_blur = self.criterion(bmap_real_B, self.blur_label_map)
        
        if warmup:
            pass

        else:
            loss_d = loss_d_sharp + loss_d_fake_sharp

        return loss_d_sharp, loss_d_fake_sharp

class ReblurLoss(nn.Module):
    def name(self):
        return 'ReblurLoss'
        
    def __init__(self):
        super(ReblurLoss,self).__init__()
        self.criterion_ssim = msssim
        self.criterion_content = nn.MSELoss()
        self.lambda_ssim = 0.1
        self.lambda_tv = 0.001
        self.lambda_reg = 0.00002

    def get_loss(self,real_B, fake_B):
        loss_content = self.criterion_content(real_B,fake_B)
        loss_ssim = 1 - self.criterion_ssim(real_B,fake_B)
        # loss_reg = torch.mean(bmap_real_B**2)
        # loss_tv = self.criterion_l1(bmap_real_B[:,:,:,:-1],bmap_real_B[:,:,:,1:]) + \
        #         self.criterion_l1(bmap_real_B[:,:,:-1,:],bmap_real_B[:,:,1:,:])

        # import ipdb; ipdb.set_trace()
        loss_reblur = self.lambda_ssim * loss_ssim \
                       + loss_content
        return loss_reblur
    
    def __call__(self, real_B, fake_B):
        return self.get_loss(real_B, fake_B)

def get_loss(loss_pick):
    if loss_pick == 'perceptual':
        loss = PerceptualLoss()
        loss.initialize(nn.MSELoss())
    elif loss_pick == 'l1':
        loss = ContentLoss()
        loss.initialize(nn.L1Loss())
    elif loss_pick == 'wgan-gp':
        loss = DiscLossWGANGP()
    # elif loss_pick == 'lsgan':
    #     loss = DiscLossLS()
    elif loss_pick == 'lsgan':
        loss = GANLoss()
    elif loss_pick == 'ragan':
        loss = RelativisticDiscLoss()
    elif loss_pick == 'ragan-ls':
        loss = RelativisticDiscLossLS()
    elif loss_pick == 'blur-gan':
        loss = BlurmapDiscLoss()
    elif loss_pick =='reblur':
        loss = ReblurLoss()
    else:
        raise ValueError("Loss [%s] not recognized." % loss_pick)
    return loss
