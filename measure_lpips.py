import numpy as np 
import math
import skimage
import skimage.io as io
import os
# from skimage.measure import lpips
import argparse
import time
import cv2
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
import lpips
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--res_root',type=str,default='/home/yjz/VFI/' ,help='the dir of restore image')
parser.add_argument('--ref_root',type=str,default='/home/yjz/datasets/Gopro_1080p/test',help='the dir of restore image')
args = parser.parse_args()
lpips_fn_alex = lpips.LPIPS(net='alex') # best forward scores

if not args.ref_root:
    args.ref_root = args.res_root

def write_txt(file_name, line):
    with open(file_name,'a') as log:
        log.write(line+'\n')
    print(line)



def LPIPS_sequence():
    print("-----calculate LPIPS for: %s ---------"%args.res_root)
    paths = sorted(os.listdir(args.res_root))
    cnt = 0.0
    t_lpips = 0
    
    record_file = os.path.dirname(args.res_root) + '/' + 'lpips.txt'
    if os.path.exists(record_file):
        os.system('rm %s'%record_file)
    if os.path.isdir(os.path.join(args.res_root,paths[0])):
        for path in paths:
            if not os.path.isdir(os.path.join(args.res_root,path)):
                continue
            res_path = os.path.join(args.res_root,path)
            res_files = sorted(os.listdir(res_path))
            res_files = [i for i in res_files if i.endswith('.png')]
            print("total %d images in directory %s"%(len(res_files),res_path))
            
            ref_path = os.path.join(args.ref_root,path,'sharp')
            ref_files = sorted(os.listdir(ref_path))
            
            ref_files = [i for i in ref_files if i.endswith('.png')]
            assert len(res_files) == len(ref_files)

            cnt_video = 0
            lpips_video = 0
            for i in range(len(res_files)):
                # import ipdb; ipdb.set_trace()
                img_name = os.path.join(res_path,res_files[i])
                img = io.imread(img_name)
                img = skimage.img_as_float32(img)

                img2_name = os.path.join(ref_path,ref_files[i])
                img2 = io.imread(img2_name)
                img2 = skimage.img_as_float32(img2)
                img2 = resize(img2, (img.shape[0], img.shape[1]))
                
                img_tensor = torch.from_numpy(img)
                img2_tensor = torch.from_numpy(img2)
                img_tensor = (img_tensor * 2 -1).permute(2,0,1).unsqueeze(0)
                img2_tensor = (img2_tensor * 2 -1).permute(2,0,1).unsqueeze(0)

                #  import ipdb; ipdb.set_trace()
                with torch.no_grad():
                    per_lpips = lpips_fn_alex.forward(img_tensor,img2_tensor)
                lpips_video += per_lpips
                cnt_video += 1
                # print('cal lpips of image %s with image %s:%.2f'%(gt_files[i], res_files[i],per_lpips))
                line = 'cal lpips of image %s :%.4f'%( res_files[i],per_lpips)
                write_txt(record_file,line)
            # print('result for video %s lpips:'%path, lpips_video / cnt_video)
            line = 'result for video %s lpips:%.4f'%(path, lpips_video / cnt_video)
            write_txt(record_file,line)

            t_lpips += lpips_video
            cnt += cnt_video
    elif os.path.isfile(os.path.join(args.res_root,paths[0])):
        
        res_files = [i for i in paths if i.endswith('fake_S.png')]
        print("total %d images in directory %s"%(len(res_files),args.res_root))
        res_files = sorted(res_files)
        
        ref_files = sorted(os.listdir(args.ref_root))
        ref_files = [i for i in ref_files if i.endswith('gtimg.png')]

        assert len(res_files) == len(ref_files)
        cnt_video = 0
        lpips_video = 0
        for i in range(len(res_files)):
            # import ipdb; ipdb.set_trace()
            img_name = os.path.join(args.res_root,res_files[i])
            img = io.imread(img_name)
            img = skimage.img_as_float32(img)

            img2_name = os.path.join(args.ref_root,ref_files[i])
            img2 = io.imread(img2_name)
            img2 = skimage.img_as_float32(img2)
            img2 = resize(img2, (img.shape[0], img.shape[1]))

            img_tensor = torch.from_numpy(img)
            img2_tensor = torch.from_numpy(img2)

            img_tensor = (img_tensor * 2 -1).permute(2,0,1).unsqueeze(0)
            img2_tensor = (img2_tensor * 2 -1).permute(2,0,1).unsqueeze(0)
            # import ipdb; ipdb.set_trace()
            
            ## tensor [-1-1]
            per_lpips = lpips_fn_alex.forward(img_tensor,img2_tensor)
            lpips_video += per_lpips
            cnt_video += 1
            # print('cal lpips of image %s with image %s:%.2f'%(gt_files[i], res_files[i],per_lpips))
            line = 'cal lpips of image %s :%.4f'%(res_files[i],per_lpips)
            write_txt(record_file,line)
        # print('result for video %s lpips:'%path, lpips_video / cnt_video)
        line = 'result for video %s lpips:%.4f'%(args.res_root, lpips_video / cnt_video)
        write_txt(record_file,line)

        t_lpips += lpips_video
        cnt += cnt_video

    # print('mean lpips:',t_lpips/cnt)
    line = '%s mean lpips:%.4f'%(args.res_root,t_lpips/cnt)
    write_txt(record_file,line)

LPIPS_sequence()