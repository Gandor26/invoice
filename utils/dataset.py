from .configs import DATA_FOLDER, IMAGE_FORMAT, SCALED_IMAGE_SIZE
from .download import download_and_convert
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

def _process_single_sample(guid, label, train, image_size, thread_storage):
    if getattr(thread_storage, 'logger', None) is None:
        thread_storage.logger = get_logger()
    try:
       box = parse_ocr_json(guid)
       original_image = imread(os.path.join(DATA_FOLDER, 'img', '{}.{}'.format(guid, IMAGE_FORMAT)), as_gray=True)
       resized_image = resize(original_image, (image_size, image_size), mode='edge', anti_aliasing=False)
    except Exception as e:
        thread_storage.logger.error('Something went wrong in parsing OCR results of {}. Skipping'.format(guid), exc_info=e)
        return
    dataset_dir = get_dir(os.path.join(DATA_FOLDER, 'set', 'train' if train else 'test', label))
    imsave(os.path.join(dataset_dir, '{}.{}'.format(guid, IMAGE_FORMAT)), resized_image)
    with open(os.path.join(dataset_dir, '{}.pkl'.format(guid)), 'wb') as f:
        pk.dump(box, f)

def _process_single_image(guid, label, train, image_size, thread_storage):
    if getattr(thread_storage, 'logger', None) is None:
        thread_storage.logger = get_logger()
    try:
        original_image = imread(os.path.join(DATA_FOLDER, 'img', '{}.{}'.format(guid, IMAGE_FORMAT)), as_gray=False)
        resized_image = resize(original_image, (image_size, image_size), mode='edge', anti_aliasing=False)
    except Exception as e:
        thread_storage.logger.error('Something went wrong in processing {}. Skipping'.format(guid), exc_info=e)
        return
    dataset_dir = get_dir(os.path.join(DATA_FOLDER, 'set', 'train' if train else 'test', label))
    imsave(os.path.join(dataset_dir, '{}.{}'.format(guid, IMAGE_FORMAT)), resized_image)

def build_dataset(*guids_and_labels, train, image_size=None):
    guids, labels = zip(*guids_and_labels)
    #download_and_convert(*guids)
    image_size = image_size or SCALED_IMAGE_SIZE
    storage = threading.local()
    Parallel(n_jobs=-1, backend='threading', verbose=False)(delayed(_process_single_image)(guid, label, train, image_size, storage)
            for guid, label in tqdm(zip(guids, labels), total=len(guids)))
