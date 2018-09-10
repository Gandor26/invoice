from utils import DATA_FOLDER, IMAGE_FORMAT
from transform import *
from loader import *
from bow import BoW
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from skimage.io import imread
from skimage.util import img_as_float
from torch import double, long
from collections import Counter
from glob import glob
import pickle as pk
import os

DEFAULT_ROOT_DIR = os.path.join(DATA_FOLDER, 'set')


class ClassificationDataset(object):
    '''
        Utility of classification dataset
        Arguments:
            dataset_type: [ImageDataset, CombinedDataset], the class of dataset
            root: root folder of dataset files
            cuda: if set to True, GPU enabled
            seed: random seed
            margin: maximum margin for padding or cropping images
            valid_split: how much of training data is used in validation
            stratified: if True, use stratified sampling for valid deataset
            threshold: only classes with at least so many samples are considered
    '''
    def __init__(self, dataset_type, root=DEFAULT_ROOT_DIR, **kwargs):
        self._train_set = dataset_type(root, training=True, **kwargs)
        self._train_idx, self._valid_idx = self._split_train_valid(**kwargs)
        self._test_set = dataset_type(root, training=False, known_classes=self._train_set.classes, **kwargs)

    def _split_train_valid(self, valid_split=0.1, stratified=True, seed=None, **kwargs):
        paths, labels = zip(*self._train_set.samples)
        if stratified:
            from sklearn.model_selection import StratifiedShuffleSplit as SSS
            splitter = SSS(n_splits=1, test_size=valid_split)
        else:
            from sklearn.model_selection import ShuffleSplit as SS
            splitter = SS(n_splits=1, test_size=valid_split)
        idx_train, idx_test = next(splitter.split(paths, labels))
        return idx_train, idx_test

    @property
    def num_classes(self):
        return self._train_set.num_classes

    def train_loader(self, batch_size, num_samples):
        self._train_set.train()
        return DataLoader(self._train_set, batch_size,
                sampler=SubsetWeightedSampler(self._train_idx, self._train_set.get_weight(self._train_idx), num_samples))

    def valid_loader(self, batch_size):
        self._train_set.eval()
        return DataLoader(self._train_set, batch_size,
                sampler=SubsetSequentialSampler(self._valid_idx))

    def test_loader(self, batch_size):
        return DataLoader(self._test_set, batch_size, shuffle=False)


class ImageDataset(DatasetFolder):
    def __init__(self, root, ext=IMAGE_FORMAT, training=True, known_classes=None, cuda=True,
            seed=None, margin=5, threshold=0, **kwargs):
        self.training = training
        self.root = os.path.join(root, 'train' if training else 'test')
        self.device = 'cuda' if cuda else 'cpu'
        self.samples, self.classes = self._make_samples(known_classes, ext, threshold)
        self.labels = LabelEncoder().fit(self.classes)
        self.augment_transform = Compose(RandomMargin(seed=seed, max_margin=margin),
                GrayscaleToTensor(device=self.device, dtype=double))
        self.normal_transform = GrayscaleToTensor(device=self.device, dtype=double)
        self.target_transform = ToTensor(device=self.device, dtype=long, transform=self.labels.transform)

    def _make_samples(self, classes, ext, threshold):
        if classes is None:
            if self.training:
                classes = list(map(lambda e:e.name, filter(lambda e: e.is_dir() and not e.name.startswith('.'),
                    os.scandir(self.root))))
            else:
                raise ValueError('Test set should be given a list of known classes')
        paths = [glob(os.path.join(self.root, cls, '*.{}'.format(ext))) for cls in classes]
        paths, classes, = zip(*filter(lambda t: len(t[0])>threshold if self.training else 0, zip(paths, classes)))
        samples = [(path, classes[i]) for i in range(len(classes)) for path in paths[i]]
        return samples, classes

    def loader(self, path):
        return img_as_float(imread(path, as_gray=True))

    def get_weight(self, indices):
        cnter = Counter([self.samples[i][1] for i in indices])
        return 1/np.array([cnter[self.samples[i][1]] for i in indices])

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def transform(self):
        return self.augment_transform if self.training else self.normal_transform

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class CombinedDataset(ImageDataset):
    def __init__(self, root, **kwargs):
        super(CombinedDataset, self).__init__(root, **kwargs)
        self.text_transform = Compose(BoW(), ToTensor(dtype=double, device=self.device))

    def _make_samples(self, classes, ext, threshold):
        image_ext = ext
        text_ext = 'pkl'
        if classes is None:
            if self.training:
                classes = list(map(lambda e:e.name, filter(lambda e: e.is_dir() and not e.name.startswith('.'),
                    os.scandir(self.root))))
            else:
                raise ValueError('Test set should be given a list of known classes')
        samples = []
        for cls in classes:
            cls_paths = []
            for image_path in glob(os.path.join(self.root, cls, '*.{}'.format(image_ext))):
                text_path = os.path.extsep.join(image_path.split(os.path.extsep)[:-1] + [text_ext])
                if os.path.exists(text_path):
                    cls_paths.append((image_path, text_path))
            if len(cls_paths) > (threshold if self.training else 0):
                samples.extend([(path_pair, cls) for path_pair in cls_paths])
        return samples, classes

    def loader(self, path_pair):
        image_path, text_path = path_pair
        image = super(CombinedDataset, self).loader(image_path)
        with open(text_path, 'rb') as f:
            box = pk.load(f)
        return image, box

    def __getitem__(self, index):
        path_pair, target = self.samples[index]
        image, box = self.loader(path_pair)
        image = self.transform(image)
        text = self.text_transform(box)
        target = self.target_transform(target)
        return (image, text), target

