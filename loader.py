from torchvision.datasets import DatasetFolder
from sklearn.preprocessing import LabelEncoder as BaseLE
from skimage.util import img_as_float, pad, crop, random_noise
from skimage.transform import resize
from skimage.io import imread
from glob import glob
from utils import DATA_FOLDER, IMAGE_FORMAT
import numpy as np
import torch as tc
import os

DEFAULT_ROOT_DIR = os.path.join(DATA_FOLDER, 'set')
DEFAULT_IMAGE_LOADER = lambda path: img_as_float(imread(path, as_gray=True))

class RandomMargin(object):
    def __init__(self, seed=42, max_margin=5):
        self.max_margin = max_margin
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
    def __init__(self, root=DEFAULT_ROOT_DIR, ext=IMAGE_FORMAT, mode='train', loader=DEFAULT_IMAGE_LOADER, cuda=True):
        self.root = os.path.join(root, mode)
        self.loader = loader
        self.device = 'cuda' if cuda else 'cpu'
        self.labels = LabelEncoder().fit(list(map(lambda e:e.name, filter(lambda e: e.is_dir() and not e.name.startswith('.'), os.scandir(self.root)))))
        self.samples = [(path, cls) for path in glob(os.path.join(self.root, cls, '*.{}'.format(ext))) for cls in self.labels.classes_]
        self.transform = GrayscaleToTensor(device=self.device, dtype=tc.double)
        self.target_transform = Compose(self.labels.transform, ToTensor(device=self.device, dtype=tc.long, transform=self.labels.transform))

class BalancedImageDataset(ImageDataset):
    def __init__(self, root=DEFAULT_ROOT_DIR, ext=IMAGE_FORMAT, mode='train', loader=DEFAULT_IMAGE_LOADER, cuda=True,
            num_sample_per_class=500):
        self.root = os.path.join(root, mode)
        self.loader = loader
        self.device = 'cuda' if cuda else 'cpu'
        self.labels = LabelEncoder().fit(list(map(lambda e:e.name, filter(lambda e: e.is_dir() and not e.name.startswith('.'), os.scandir(self.root)))))
        self.num_sample_per_class = num_sample_per_class
        self.samples = []
        for cls in self.labels.classes_:
            files = glob(os.path.join(self.root, cls, '*.{}'.format(ext)))
            if len(files) < self.num_sample_per_class:
                self.samples.extend([(f, cls) for f in np.random.choice(files, self.num_sample_per_class, replace=True)])
            elif len(files) > self.num_sample_per_class:
                self.samples.extend([(f, cls) for f in np.random.choice(files, self.num_sample_per_class, replace=False)])
            else:
                self.samples.extend([(f, cls) for f in files])
        self.transform = Compose(RandomMargin(max_margin=8), GrayscaleToTensor(device=self.device, dtype=tc.double))
        self.target_transform = Compose(self.labels.transform, ToTensor(device=self.device, dtype=tc.long))

Dataset = BalancedImageDataset
