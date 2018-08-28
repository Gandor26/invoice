from .configs import DATA_FOLDER, IMAGE_FORMAT, SCALED_IMAGE_SIZE
from .database import get_labels
from .textbox import parse_ocr_json
from .misc import get_dir, get_logger
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle as pk
import threading
import os

__all__ = ['build_dataset']

def _process_single_sample(guid, label, train, thread_storage):
    if getattr(thread_storage, 'logger', None) is None:
        thread_storage.logger = get_logger()
    try:
        box = parse_ocr_json(guid, train)
        image = imread(os.path.join(DATA_FOLDER, 'img', 'train' if train else 'test', '{}.{}'.format(guid, IMAGE_FORMAT)), as_gray=True)
        resized_image = resize(image, (SCALED_IMAGE_SIZE, SCALED_IMAGE_SIZE))
    except:
        thread_storage.logger.warn('Something wrong in parsing OCR results of {}. Skipping'.format(guid))
        return
    dataset_dir = get_dir(os.path.join(DATA_FOLDER, 'set', 'train' if train else 'test', label))
    imsave(os.path.join(dataset_dir, '{}.{}'.format(guid, IMAGE_FORMAT)), resized_image)
    with open(os.path.join(dataset_dir, '{}.pkl'.format(guid)), 'wb') as f:
        pk.dump(box, f)

def build_dataset(train=True):
    files = glob(os.path.join(DATA_FOLDER, 'img', 'train' if train else 'test', '*.{}'.format(IMAGE_FORMAT)))
    guids = list(map(lambda s: os.path.split(f)[-1].split(os.path.extsep)[0], files))
    labels = list(map(lambda t: '{}_{}'.join(t[0], t[1]), get_labels(*guids, vendor=True, flatten=True)))
    storage = threading.local()
    Parallel(n_jobs=-1, backend='threading', verbose=False)(delayed(_process_single_sample)(guid, label, train, storage)
            for guid, label in tqdm(zip(guids, labels), total=len(guids)))
