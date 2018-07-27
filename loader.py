from torch.utils import data as tdata
from skimage.io import imread, imshow, imsave
from skimage.transfrom import rotate, resize
from sklearn.preprocessing import LabelEncoder
from glob import glob
import itertools
import torch as tc
import os

DEFAULT_IMAGE_SIZE=300

def load_image(path, rot, size=DEFAULT_IMAGE_SIZE):
    image = imread(path, as_gray=True)
    image = rotate(image, rot)
    image = resize(image, (size, size))
    return image

class ImageFolder(tdata.Dataset):
    def __init__(self, root_dir, cuda=False):
        super(ImageFolder, self).__init__()
        self.root = root_dir
        self.classes = LabelEncoder().fit([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        self.samples = sum([(path, path.split(os.path.sep)[-2], rot) for rot in [0,90,180,270]] for path in glob(self.root, '*', '*.png'))
        self.cuda = cuda

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, cls, rot = self.samples[index]
        tensor = tc.from_numpy(load_image(path, rot)).float().div_(255).unsqueeze_(0)
        label = tc.from_numpy(self.classes.transform([cls])).long()
        if self.cuda:
            tensor = tensor.cuda()
            label  = label.cuda()
        return tensor, label

Dataset = ImageFolder
