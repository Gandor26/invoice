from torchvision.datasets import DatasetFolder
from torch.utils.data import Sampler
from sklearn.preprocessing import LabelEncoder as BaseLE
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.model_selection import ShuffleSplit as SS
from skimage.util import img_as_float, pad, crop, random_noise
from skimage.transform import resize
from skimage.io import imread
from collections import Counter
from glob import glob
from utils import DATA_FOLDER, IMAGE_FORMAT
import numpy as np
import torch as tc
import os

DEFAULT_ROOT_DIR = os.path.join(DATA_FOLDER, 'set')
DEFAULT_IMAGE_LOADER = lambda path: img_as_float(imread(path, as_gray=True))

class RandomMargin(object):
    def __init__(self, seed=None, max_margin=5):
        self.max_margin = max_margin
        if seed is not None:
            np.random.seed(seed)
    def __call__(self, image):
        pad_or_crop = np.random.randint(2)
        if pad_or_crop:
            # pad
            margin_top, margin_bottom, margin_left, margin_right  = np.random.randint(self.max_margin+1, size=4)
            padded_image = pad(image, ((margin_top, margin_bottom), (margin_left, margin_right)), mode='constant', constant_values=1.0)
            resized_image = resize(padded_image, image.shape)
        else:
            # crop
            margin_top, margin_bottom, margin_left, margin_right  = np.random.randint(self.max_margin+1, size=4)
            cropped_image = crop(image, ((margin_top, margin_bottom), (margin_left, margin_right)), copy=True)
            resized_image = resize(cropped_image, image.shape)
        return resized_image

class Compose(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, vals):
        for t in self.transforms:
            vals = t(vals)
        return vals

class LabelEncoder(BaseLE):
    def fit(self, y):
        if np.isscalar(y):
            y = [y]
        return super(LabelEncoder, self).fit(y)
    def fit_transform(self, y):
        wrapped = [y] if np.isscalar(y) else y
        wrapped = super(LabelEncoder, self).fit_transform(wrapped)
        return wrapped.item() if np.isscalar(y) else wrapped
    def transform(self, y):
        wrapped = [y] if np.isscalar(y) else y
        wrapped = super(LabelEncoder, self).transform(wrapped)
        return wrapped.item() if np.isscalar(y) else wrapped
    def inverse_transform(self, y):
        wrapped = [y] if np.isscalar(y) else y
        wrapped = super(LabelEncoder, self).inverse_transform(wrapped)
        return wrapped.item() if np.isscalar(y) else wrapped

class ToTensor(object):
    def __init__(self, device, dtype, transform=None):
        self.device = device
        self.dtype = dtype
        self.transform = transform
    def __call__(self, vals):
        if self.transform is not None:
            vals = self.transform(vals)
        return tc.as_tensor(vals, device=self.device, dtype=self.dtype)

class GrayscaleToTensor(ToTensor):
    def __init__(self, device, dtype):
        super(GrayscaleToTensor, self).__init__(device, dtype, transform=lambda image: image[np.newaxis])

class ImageDataset(DatasetFolder):
    def __init__(self, root=DEFAULT_ROOT_DIR, ext=IMAGE_FORMAT, mode='train', loader=DEFAULT_IMAGE_LOADER, cuda=True,
            seed=None, margin=5, threshold=0):
        self.training = True if mode == 'train' else False
        self.root = os.path.join(root, mode)
        self.loader = loader
        self.device = 'cuda' if cuda else 'cpu'
        self.classes = list(map(lambda e:e.name, filter(lambda e: e.is_dir() and not e.name.startswith('.'), os.scandir(self.root))))
        self.paths = [glob(os.path.join(self.root, cls, '*.{}'.format(ext))) for cls in self.classes]
        self.paths, self.classes, = zip(*filter(lambda t: len(t[0])>threshold, zip(self.paths, self.classes)))
        self.labels = LabelEncoder().fit(self.classes)
        self.samples = [(path, self.classes[i]) for i in range(len(self.classes)) for path in self.paths[i]]
        self.augment_transform = Compose(RandomMargin(seed=seed, max_margin=margin), GrayscaleToTensor(device=self.device, dtype=tc.double))
        self.normal_transform = GrayscaleToTensor(device=self.device, dtype=tc.double)
        self.target_transform = ToTensor(device=self.device, dtype=tc.long, transform=self.labels.transform)

    def get_weight(self, indices):
        cnter = Counter([self.samples[i][1] for i in indices])
        return 1/np.array([cnter[self.samples[i][1]] for i in indices])

    @property
    def transform(self):
        return self.augment_transform if self.training else self.normal_transform

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class SubsetWeightedSampler(Sampler):
    def __init__(self, indices, weights, num_samples):
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = tc.as_tensor(weights, dtype=tc.double)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        idx = tc.multinomial(self.weights, self.num_samples, replacement=True)
        return (self.indices[i] for i in idx)

def split_train_valid(dataset, valid_split, stratified=True, seed=None):
    paths, labels = zip(*dataset.samples)
    Splitter = SSS if stratified else SS
    splitter = Splitter(n_splits=1, test_size=valid_split)
    idx_train, idx_test = next(splitter.split(paths, labels))
    return idx_train, idx_test
