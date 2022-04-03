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

class AlignedDataset(data.Dataset):
    def __init__(self, config, train):
        # super(AlignedDataset,self).__init__(config)
        self.config = config
        self.train = train
        self.root = config['blurry_videos']
        if train:
            self.dir_AB = os.path.join(self.root, 'train') 
        else:
            self.dir_AB = os.path.join(self.root, 'test') 

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        # we use more data augmentation
        # if config['phase'] == 'train':
        #     multiframe_data = ['9frame']
        #     dir_root = os.path.dirname(self.root)
        #     for i in multiframe_data:
        #         dir_multiframe = os.path.join(dir_root,"gopro_%s"%i,config['phase'])
        #         multiframe_paths = sorted(make_dataset(dir_multiframe))
        #         self.AB_paths += multiframe_paths
        #assert(config['resize_or_crop'] == 'resize_and_crop')

        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5),
        #                                        (0.5, 0.5, 0.5))]   
        
        transform_list = [#transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                            transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # data aug with resize
        # input_sizeX,input_sizeY = AB.size[0],AB.size[1] 
        # if self.config['phase'] =='train':
        #     # if random.random() < 0.5:
        #     resize = random.uniform(1,1.5) 
        #     AB = AB.resize((int(input_sizeX * resize), int(input_sizeY*resize)), Image.BICUBIC)
            
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.config['crop_size_X'] - 1))
        h_offset = random.randint(0, max(0, h - self.config['crop_size_Y'] - 1))

        A = AB[:, h_offset:h_offset + self.config['crop_size_Y'],
               w_offset:w_offset + self.config['crop_size_X']]
        B = AB[:, h_offset:h_offset + self.config['crop_size_Y'],
               w + w_offset:w + w_offset + self.config['crop_size_X']]

        # if not crop in test time
        if self.train:
            if random.random() < 0.5:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(2, idx)
                B = B.index_select(2, idx)
            if random.random() < 0.5:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(1, idx)
                B = B.index_select(1, idx)
        else:
            A = AB[:,:,:w]
            B = AB[:,:,w:w_total]
        return {'B': A, 'S': B,
                'B_path': AB_path, 'gt':True}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
