import torch 
import numpy as np 
import os
from PIL import Image
import cv2

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().detach().numpy()
    if len(image_numpy.shape) == 3:
        image_numpy = np.clip(np.transpose(image_numpy, (1, 2, 0)),0,1)  * 255.0
    elif len(image_numpy.shape) == 4:
        image_numpy = np.clip(np.transpose(image_numpy, (0, 2, 3, 1)),0,1)  * 255.0
   
    return image_numpy.astype(imtype)

def load_image(filename, trans_list=None, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    
    if trans_list:
        img = trans_list(img)
    img = img.unsqueeze(0)
    return img


def save_image(image_numpy, image_path):
    image_pil = None
    if image_numpy.shape[2] == 1:
        image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
        image_pil = Image.fromarray(image_numpy, 'L')
    else:
        image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_train_sample(config,epoch, results):
    save_dir = os.path.join(config['checkpoints'], config['model_name'], 'sample')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for label, image_numpy in results.items():
        save_path = os.path.join(save_dir, 'epoch%.3d_%s.png'%(epoch,label))
        save_image(image_numpy[0], save_path)

def save_test_images(config, save_dir, results, image_path):
    for i in range(len(image_path)):
        B_path = image_path['B_path'][0]
        others, B_name = os.path.split(B_path)
        if os.path.split(others)[-1] == 'blur':
            video_name = others.split('/')[-2]
        else:
            video_name = os.path.split(others)[-1]


        video_dir = os.path.join(save_dir, video_name)
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        
        frame_index_B = B_name.split('.')[0]

        # save image to corespound path
        save_B_path = os.path.join(video_dir,"%s_real_B.png"%(frame_index_B))
        save_S_path = os.path.join(video_dir,"%s_real_S.png"%(frame_index_B))
        save_fakeS_path = os.path.join(video_dir,"%s_fake_S.png"%(frame_index_B))
        save_fakeB_path = os.path.join(video_dir,"%s_fake_B.png"%(frame_index_B))
        save_image(results['real_B'][i],save_B_path)
        save_image(results['real_S'][i],save_S_path)
        save_image(results['fake_S'][i],save_fakeS_path)
        save_image(results['fake_B'][i],save_fakeB_path)

def save_heat_mmap(mmap, image_path):
    heatmap = mmap2heat(mmap)
    save_image(heatmap, image_path)

def mmap2heat(mmap_gpu):
    mmap = mmap_gpu.cpu()
    mmap = mmap.detach().numpy()
    mmap = np.squeeze(mmap)

    mmap = np.transpose(mmap,(1,2,0))
    H,W,C = mmap.shape
    
    vec = mmap 
    hsv = np.zeros((H,W,3),dtype=np.uint8)
    hsv[...,2] = 255

    # # method1: vector sum
    index = np.where(vec[...,1] < 0)
    vec[index] = -vec[index]  
    mag,ang = cv2.cartToPolar(vec[...,1], -vec[...,0])
    hsv[...,0] = ang * 180 / np.pi / 2
    # print("max:",mag.max(),"min",mag.min())
    # mag[-1,-1] = 0.25
    hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def print_config(config):
    for key,value in config.items():
        if isinstance(value, dict):
            print('\n {:s}:'.format(key))
            print_config(value)
        else:
            print('%s:%s'%(key,value))

def parse(parser):

    opt = parser.parse_config()

    str_ids = opt.gpu.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)

    # set gpu ids
    # if len(opt.gpu) > 0:
    #     torch.cuda.set_device(opt.gpu[0])

    config = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(config.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints, opt.model_name)
    os.makedirs(expr_dir,exist_ok=True)
    if opt.train:
        phase = "train"
    else:
        phase = "test"
    file_name = os.path.join(expr_dir, 'opt_%s.txt'%phase)
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(config.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
    return opt