import scipy.io as sio 
import os
import cv2
from utils import utils
from glob import glob 
import numpy as np 

def bmap2heat(bmap):
    bmap = np.squeeze(bmap)

    H,W,C = bmap.shape
    
    vec = bmap 
    hsv = np.zeros((H,W,3),dtype=np.uint8)
    hsv[...,2] = 255

    # # method1: vector sum
    # index = np.where(vec[...,1] < 0)
    # vec[index] = -vec[index]  
    
    # import ipdb; ipdb.set_trace()
    vec = vec.astype('float64')
    mag,ang = cv2.cartToPolar(vec[...,0], -vec[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    # print("max:",mag.max(),"min",mag.min())
    mag[-1,-1] = 0
    hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr


root = '/home/yjz/datasets/Synthetic_blur/test_small/'
path_pattern = '/home/yjz/datasets/Synthetic_blur/test_small/*_mfmap.mat'
dst_root = 'exp_results/Mmap_vis'
if not os.path.exists(dst_root):
    os.mkdir(dst_root)

mmap_files = sorted(glob(path_pattern))
for file_i in mmap_files:
    name = os.path.basename(file_i)
    name = os.path.splitext(name)[0]
    mmap = sio.loadmat(file_i)['mfmap']
    heat = bmap2heat(mmap)
    out_path = os.path.join(dst_root,"%s.png"%name)
    cv2.imwrite(out_path,heat)
    # import ipdb; ipdb.set_trace()

