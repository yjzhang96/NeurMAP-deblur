## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808

import os
import ipdb
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
import argparse
import lpips
import torch

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--gt_dir', default='/hdd/deblur_datasets/RealBlur/RealBlur-J_dataset/test/target/*.png', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
args = parser.parse_args()
lpips_fn_alex = lpips.LPIPS(net='alex') # best forward scores

def write_txt(file_name, line):
    with open(file_name,'a') as log:
        log.write(line+'\n')
    print(line)


def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift

def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def proc(filename):
    tar,prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0
    
    
    prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)

    img_tensor = torch.from_numpy(prd_img)
    img2_tensor = torch.from_numpy(tar_img)
    img_tensor = (img_tensor * 2 -1).permute(2,0,1).unsqueeze(0)
    img2_tensor = (img2_tensor * 2 -1).permute(2,0,1).unsqueeze(0)

    #  import ipdb; ipdb.set_trace()
    with torch.no_grad():
        per_lpips = lpips_fn_alex.forward(img_tensor,img2_tensor)

    PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
    SSIM = compute_ssim(tar_img, prd_img, cr1)
    return (PSNR,SSIM,per_lpips)

datasets = ['RealBlur_J', 'RealBlur_R']

# for dataset in datasets:

#     file_path = os.path.join('results' , dataset)
#     gt_path = os.path.join('Datasets', dataset, 'test', 'target')

#     path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
#     gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))

#     assert len(path_list) != 0, "Predicted files not found"
#     assert len(gt_list) != 0, "Target files not found"

#     psnr, ssim = [], []
#     img_files =[(i, j) for i,j in zip(gt_list,path_list)]
#     with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
#         for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
#             psnr.append(PSNR_SSIM[0])
#             ssim.append(PSNR_SSIM[1])

#     avg_psnr = sum(psnr)/len(psnr)
#     avg_ssim = sum(ssim)/len(ssim)

#     print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))


gt_path = args.gt_dir
file_path = args.result_dir
record_file = os.path.dirname(file_path) + 'RealBlur_metric.txt'
if os.path.exists(record_file):
  os.system('rm %s'%record_file)
  
path_list = natsorted(glob(file_path))
gt_list = natsorted(glob(gt_path))

assert len(path_list) != 0, "Predicted files not found"
assert len(gt_list) != 0, "Target files not found"
assert len(path_list) == len(gt_list), "result files %d not correspond with gt files %d"%(len(path_list),len(gt_list))

psnr, ssim,  = [], []
t_lpips = 0
img_files =[(i, j) for i,j in zip(gt_list,path_list)]
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
        psnr.append(PSNR_SSIM[0])
        ssim.append(PSNR_SSIM[1])
        t_lpips += PSNR_SSIM[2].numpy()
        line = 'file %s PSNR: %.2f \t SSIM:%.4f \t LPIPS:%.4f'%(filename[1],PSNR_SSIM[0],PSNR_SSIM[1],PSNR_SSIM[2])
        write_txt(record_file, line)
avg_psnr = sum(psnr)/len(psnr)
avg_ssim = sum(ssim)/len(ssim)
avg_lpips = t_lpips/len(ssim)

line = '\n For %s dataset PSNR: %.2f SSIM: %.4f \t LPIPS: %.4f'%(file_path, avg_psnr, avg_ssim, avg_lpips)
write_txt(record_file, line)