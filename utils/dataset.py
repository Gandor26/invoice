from glob import glob
from skimage.io import imread, imshow, imsave
from skimage.transform import rotate, resize
from .database import get_labels
from .download import DATA_FOLDER, IMAGE_FORMAT
import os

#DATA_FOLDER = os.path.expanduser('~/workspace/invoice/data')
#IMAGE_FORMAT = 'png'

def process_image():
    pass

def split_dataset():
    pass

def build_dataset(data_folder=DATA_FOLDER, train=True, image_format=IMAGE_FORMAT):
    image_files = glob(os.path.join(data_folder, 'img', 'train' if train else 'test', '*.png'))
