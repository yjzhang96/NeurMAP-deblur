# NeurMAP-deblur (official pytorch implementation)
[[paper]]() [[video]]()

This repository provides the official PyTorch implementation of the paper:

>NeurMAP: Neural Maximum A Posteriori Estimation on Unpaired Data for Motion Deblurring
> 
>Youjian Zhang, Chaoyue Wang, Dacheng Tao
>
>Abstract: Real-world dynamic scene deblurring has long been a challenging task since paired blurry-sharp training data is unavailable. Conventional Maximum A Posteriori estimation and deep learning-based deblurring methods are restricted by handcrafted priors and synthetic blurry-sharp training pairs respectively, thereby failing to generalize to real dynamic blurriness. To this end, we propose a Neural Maximum A Posteriori (NeurMAP) estimation framework for training neural networks to recover blind motion information and sharp content from unpaired data. The proposed NeruMAP consists of a motion estimation network and a deblurring network which are trained jointly to model the (re)blurring process (i.e. likelihood function). Meanwhile, the motion estimation network is trained to explore the motion information in images by applying implicit dynamic motion prior, and in return enforces the deblurring network training (i.e. providing sharp image prior). The proposed NeurMAP is an orthogonal approach to existing deblurring neural networks, and is the first framework that enables training image deblurring networks on unpaired datasets. Experiments demonstrate our superiority on both quantitative metrics and visual quality over state-of-the-art methods. 

<img src= "xx" width="90%">

---
## Contents

The contents of this repository are as follows:

1. [Prerequisites](#Prerequisites)
2. [Dataset](#Dataset)
3. [Train](#Train)
4. [Test](#Test)
5. [Performance](#Performance)
6. [Model](#Model)

---

### Prerequisites
- Pytorch 1.1.0 + cuda 10.0
- You need to first install two repositories, [DCN_v2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) and [MSSSIM](https://github.com/jorge-pessoa/pytorch-msssim), in the './model' directory, following their installation instructions respectively.
### Dataset
Download [GoPro]((https://seungjunnah.github.io/Datasets/gopro.html)) datasets and algin the blurry/sharp image pairs.
Organize the dataset in the following form:

```bash
|- Gopro_align_data 
|   |- train  % 2103 image pairs
|   |   |- GOPR0372_07_00_000047.png
|   |   |- ...
|   |- test   % 1111 image pairs
|   |   |- GOPR0384_11_00_000001.png
|   |   |- ...
```

### Training 
- To train motion offset estimation model, run the following command:
```bash
sh run_semi_train.sh
```


### Test
- To train motion offset estimation model, run the following command:
```bash
python test.py --config=configs/config_test.yaml
```


### Performance
- to be updated

### Model
We have put the pretrained model in the Google drive.

|   Dataset     |     Model  | |
| :---------: |     :-------:       | :-------:  |
|     kernel-synthesized    |     [DMPHN](https://drive.google.com/drive/u/0/folders/1keoykuKd4-aLrhFrd6P_WnwYr0br-AUh)          |   [MPRNet](https://drive.google.com/drive/u/0/folders/1keoykuKd4-aLrhFrd6P_WnwYr0br-AUh)       |
|    RealBlur     |     [DMPHN](https://drive.google.com/drive/u/0/folders/1keoykuKd4-aLrhFrd6P_WnwYr0br-AUh)          |   [MPRNet](https://drive.google.com/drive/u/0/folders/1keoykuKd4-aLrhFrd6P_WnwYr0br-AUh)       |  



