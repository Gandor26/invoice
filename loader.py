from torch.utils import data as tdata
from skimage.io import imread, imshow, imsave
from skimage.transform import rotate, resize
from sklearn.preprocessing import LabelEncoder
from glob import glob
import itertools
import torch as tc
import os

DEFAULT_IMAGE_SIZE=224

def load_image(path, rot, size=DEFAULT_IMAGE_SIZE):
    image = imread(path, as_gray=True)
    #image = resize(image, (size, size))
    image = rotate(image, rot)
    return image

class ImageFolder(tdata.Dataset):
    def __init__(self, root_dir, cuda=False):
        super(ImageFolder, self).__init__()
        self.root = root_dir
        self.classes = LabelEncoder().fit([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))\
                                          and not d.startswith('.')])
        self.samples = sum([[(path, path.split(os.path.sep)[-2], rot) for rot in [0]] for path in glob(os.path.join(self.root, '*', '*.png'))], [])
        self.cuda = cuda

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, cls, rot = self.samples[index]
        tensor = tc.from_numpy(load_image(path, rot)).float().unsqueeze_(0)
        label = tc.from_numpy(self.classes.transform([cls])).long().squeeze_(0)
        if self.cuda:
            tensor = tensor.cuda()
            label  = label.cuda()
        return tensor, label

Dataset = ImageFolder
