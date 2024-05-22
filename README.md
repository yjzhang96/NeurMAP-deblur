# NeurMAP-deblur (official pytorch implementation)
[[paper]](https://arxiv.org/abs/2204.12139) 

This repository provides the official PyTorch implementation of the paper:

>NeurMAP: Neural Maximum A Posteriori Estimation on Unpaired Data for Motion Deblurring
> 
>Youjian Zhang, Chaoyue Wang, Dacheng Tao
>
>Abstract: Real-world dynamic scene deblurring has long been a challenging task since paired blurry-sharp training data is unavailable. Conventional Maximum A Posteriori estimation and deep learning-based deblurring methods are restricted by handcrafted priors and synthetic blurry-sharp training pairs respectively, thereby failing to generalize to real dynamic blurriness. To this end, we propose a Neural Maximum A Posteriori (NeurMAP) estimation framework for training neural networks to recover blind motion information and sharp content from unpaired data. The proposed NeruMAP consists of a motion estimation network and a deblurring network which are trained jointly to model the (re)blurring process (i.e. likelihood function). Meanwhile, the motion estimation network is trained to explore the motion information in images by applying implicit dynamic motion prior, and in return enforces the deblurring network training (i.e. providing sharp image prior). The proposed NeurMAP is an orthogonal approach to existing deblurring neural networks, and is the first framework that enables training image deblurring networks on unpaired datasets. Experiments demonstrate our superiority on both quantitative metrics and visual quality over state-of-the-art methods. 

<img src= "https://github.com/yjzhang96/NeurMAP-deblur/blob/main/pipeline.jpg" width="90%">

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
#### Environment
- Pytorch 1.1.0 + cuda 10.0
- You need to first install two repositories, [DCN_v2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) and [MSSSIM](https://github.com/jorge-pessoa/pytorch-msssim), in the './model' directory, following their installation instructions respectively.
#### Backbone model
- You need to prepare the backbone model for finetuning. In our experiments, we have tried [DMPHN](https://github.com/HongguangZhang/DMPHN-cvpr19-master), [MPRNet](https://github.com/swz30/MPRNet), and [MIMO-UNet](https://github.com/chosj95/MIMO-UNet). For your convenience, we have upload corresponding model weights in [Google Drive](https://drive.google.com/drive/u/0/folders/1-bJ--advZpZh_G6-XDZkn_q46w9hBkSz). Feel free to download them and place them in folder ```pretrain_models```. 
- Also, we have provide a pretrained motion offset estimation model in [Google Drive](https://drive.google.com/drive/u/0/folders/1QKzgZc6hHZ7qMMgPJDqAhqFDEY9DI9HV). You can organize the pretrained models as following:
```
|- pretrain_models
|   |- MPRnet
|   |   |- model_deblurring.pth
|   |- DMPHN
|   |   |- G_net_latest.pth
|   |- MTR
|   |   |- lin
|   |   |   |- latest_net_offset.pth
```

### Dataset
You will need both GoPro dataset (paired dataset) and an unpaired dataset to conduct the semi-supervised training. If you want to conduct unsupervised training, just prepare an unpaired dataset. 

For GoPro dataset, Download from [data]((https://seungjunnah.github.io/Datasets/gopro.html)). Organize the dataset in the following form:

```bash
|- Gopro_dataset 
|   |- train  % 2103 image pairs
|   |   |- GOPR0372_07_00_000047.png
|   |   |- ...
|   |- test   % 1111 image pairs
|   |   |- GOPR0384_11_00_000001.png
|   |   |- ...
```

### Training 
- To train all the modules in a semi-supervised way, run the following command:
```bash
sh run_semi_train.sh
```


### Test
- To test the model, run the following command:
```bash
python test.py --config=configs/config_test.yaml
```


### Model
We have put the pretrained models in the Google drive.

|   Dataset     |     Model  | |
| :---------: |     :-------:       | :-------:  |
|     kernel-synthesized    |     [DMPHN](https://drive.google.com/drive/u/0/folders/1hZOEkRAJtMyB8cynPtqiOePjPa4LscxH)          |   [MPRNet](https://drive.google.com/drive/u/0/folders/15851Yn71pJyr6zoE2ysH7xyW_op4O92-)       |
|    RealBlur     |     [DMPHN](https://drive.google.com/drive/u/0/folders/18EY3kYOMwUATT2jfocGHhl5BnHC97epO)          |   [MPRNet](https://drive.google.com/drive/u/0/folders/1nr1-XdA4KQp15lmZWEJeBuPiGyBxLf6k)       |  



