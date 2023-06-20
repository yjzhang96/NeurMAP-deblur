import torch.utils.data as data
from PIL import Image
import os
import os.path
from glob import glob
import random
import torchvision.transforms as transforms

def _make_dataset(blurry_dir,deblur_dir,config):
    """
    Creates a 2D list of all the frames in N clips containing
    M frames each.

    2D List Structure:
    [[frame00, frame01,...frameM]  <-- clip0
     [frame00, frame01,...frameM]  <-- clip0
     :
     [frame00, frame01,...frameM]] <-- clipN

    Parameters
    ----------
        dir : string
            root directory containing clips.

    Returns
    -------
        list
            2D list described above.
    """


    
    
    framesPath = []
    # Find and loop over all the clips in root `dir`.
    count = 0
    blurry_img_paths = sorted(glob(blurry_dir, recursive=True))
    deblur_img_paths = sorted(glob(deblur_dir, recursive=True))
    assert len(blurry_img_paths) == len(deblur_img_paths)
    for index in range(len(blurry_img_paths)):
        
        framesPath.append({})
        framesPath[count]['B'] = blurry_img_paths[index]
        framesPath[count]['S'] = deblur_img_paths[index]
        count += 1
    return framesPath


def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    """
    Opens image at `path` using pil and applies data augmentation.

    Parameters
    ----------
        path : string
            path of the image.
        cropArea : tuple, configional
            coordinates for cropping image. Default: None
        resizeDim : tuple, configional
            dimensions for resizing image. Default: None
        frameFlip : int, configional
            Non zero to flip image horizontally. Default: 0

    Returns
    -------
        list
            2D list described above.
    """


    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = resized_img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')
    
    
class BlurryVideo(data.Dataset):
    """
    A dataloader for loading N samples arranged in this way:

        |-- video0
            |-- frameB0 frameB1 -- frameB0_S frameB0_E frameB1_S frameB1_E
            |-- frame01
            :
            |-- framexx
            |-- frame12
        |-- clip1
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12
        :
        :
        |-- clipN
            |-- frame00
            |-- frame01
            :
            |-- frame11
            |-- frame12

    ...

    Attributes
    ----------
    framesPath : list
        List of frames' path in the dataset.

    Methods
    -------
    __getitem__(index)
        Returns the sample corresponding to `index` from dataset.
    __len__()
        Returns the size of dataset. Invoked as len(datasetObj).
    __repr__()
        Returns printable representation of the dataset object.
    """


    def __init__(self, config, train):
        """
        Parameters
        ----------
            root : string
                Root directory path.
            transform : callable, configional
                A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            dim : tuple, configional
                Dimensions of images in dataset. Default: (640, 360)
            randomCropSize : tuple, configional
                Dimensions of random crop to be applied. Default: (352, 352)
            train : boolean, configional
                Specifies if the dataset is for training or testing/validation.
                `True` returns samples with data augmentation like random 
                flipping, random cropping, etc. while `False` returns the
                samples without randomization. Default: True
        """


        # Populate the list with image paths for all the
        # frame in `root`.
        self.config = config
        if train:
            self.blurry_dir = config['train']['blur_videos']
            self.deblur_dir = config['train']['deblur_videos']
            
        else:
            if config['is_training']:
                self.blurry_dir = config['val']['blur_videos']
                self.deblur_dir = config['val']['deblur_videos']
            else:
                self.blurry_dir = config['test']['blur_videos']
                self.deblur_dir = config['test']['deblur_videos']

           
        framesPath = _make_dataset(self.blurry_dir,self.deblur_dir,config)

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: datasets"))
                
        self.dim = (1280,720)
        
        self.framesPath     = framesPath
        self.train = train
        # mean = [0.5,0.5,0.5]
        # std = [1,1,1]
        # normalize = transforms.Normalize(mean=mean,
        #                                 std = std)
        if train:
            # random_crop = transforms.RandomCrop(config['crop_size_X'],config['crop_size_Y'])
            self.transform = transforms.Compose([transforms.ToTensor() ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Returns the sample corresponding to `index` from dataset.

        The sample consists of two reference frames - B1 and B2 -
        and coresponding start and end frame groundtruth B1_S B1_E ... 

        Parameters
        ----------
            index : int
                Index

        Returns
        -------
            tuple
                (sample, returnIndex) where sample is 
                [I0, intermediate_frame, I1] and returnIndex is 
                the position of `random_intermediate_frame`. 
                e.g.- `returnIndex` of frame next to I0 would be 0 and
                frame before I1 would be 6.
        """


        sample = {}
        
        if (self.train):
            ### Data Augmentation ###
            # random scale from 1.0 to 2.0
            # random_scale = random.random() + 1.0
            dim = self.dim
            # import ipdb; ipdb.set_trace()
            # Apply random crop on the input frames
            self.cropX0         = dim[0] - self.config['crop_size_X']
            self.cropY0         = dim[1] - self.config['crop_size_Y']
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.config['crop_size_X'], cropY + self.config['crop_size_Y'])

                
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            cropArea = None
            randomFrameFlip = 0
            dim = self.dim
            if not self.config['is_training']:
                dim = None
        # Loop over for all frames corresponding to the `index`.
        for key, path in self.framesPath[index].items():
            # Open image using pil and augment the image.
            image = _pil_loader(path, cropArea=cropArea, resizeDim=dim ,frameFlip=randomFrameFlip)
            
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample[key] = image
        sample['B_path'] = self.framesPath[index]['B']
        sample['gt'] = True
        return sample


    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).

        Returns
        -------
            int
                number of samples.
        """


        return len(self.framesPath)

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """


        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.blurry_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
