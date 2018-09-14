from sklearn.preprocessing import LabelEncoder as BaseLE
from PIL import Image, ImageOps
import numpy as np
import torch as tc

class Compose(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, vals):
        for t in self.transforms:
            vals = t(vals)
        return vals

    def __getitem__(self, index):
        return self.transforms[index]


class RandomMargin(object):
    def __init__(self, max_margin=10, seed=None):
        np.random.seed(seed)
        self.max_margin = max_margin

    def _crop(self, image):
        margin_top, margin_bottom, margin_left, margin_right = np.random.randint(self.max_margin+1, size=4)
        cropped_image = ImageOps.crop(image, (margin_left, margin_top, margin_right, margin_bottom))
        resized_image = cropped_image.resize(image.size, resample=Image.BILINEAR)
        return resized_image

    def _pad(self, image):
        margin_top, margin_bottom, margin_left, margin_right = np.random.randint(self.max_margin+1, size=4)
        padded_image = ImageOps.expand(image, (margin_left, margin_top, margin_right, margin_bottom), fill=255)
        resized_image = padded_image.resize(image.size, resample=Image.BILINEAR)
        return resized_image

    def __call__(self, image):
        pad_or_crop = np.random.randint(2)
        return self._pad(image) if pad_or_crop else self._crop(image)


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
        def transform(image):
            array = np.array(image, dtype=np.uint8)/255.0
            return array[np.newaxis]
        super(GrayscaleToTensor, self).__init__(device, dtype, transform=transform)

