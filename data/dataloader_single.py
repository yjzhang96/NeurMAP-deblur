import os.path
import random
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if "map" in dir:
        for fname in sorted(os.listdir(dir)):
            path = os.path.join(dir,fname)
            images.append(path)
    else:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

class SingleDataset(data.Dataset):
    def __init__(self, config, train):
        # super(AlignedDataset,self).__init__(config)
        self.config = config
        self.train = train
        self.root = config.dataroot
        self.img_paths = sorted(make_dataset(self.root))

        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5),
        #                                        (0.5, 0.5, 0.5))]   
        
        transform_list = [#transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                            transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')

        # data aug with resize
        # input_sizeX,input_sizeY = img.size[0],img.size[1] 
        # if self.config['phase'] =='train':
        #     # if random.random() < 0.5:
        #     resize = random.uniform(1,1.5) 
        #     img = img.resize((int(input_sizeX * resize), int(input_sizeY*resize)), Image.BICUBIC)
            
        img = self.transform(img)
        # if not crop in test time

        return {'input': img, 
                'img_path': img_path}

    def __len__(self):
        return len(self.img_paths)

    def name(self):
        return 'SingleDataset'
