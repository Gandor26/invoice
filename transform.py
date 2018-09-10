from sklearn.preprocessing import LabelEncoder as BaseLE
from skimage.util import pad, crop, random_noise
from skimage.transform import resize
import numpy as np
import torch as tc

class Compose(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, vals):
        for t in self.transforms:
            vals = t(vals)
        return vals

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

