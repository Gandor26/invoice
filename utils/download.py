from .database import get_labels
from .misc import get_dir, get_logger
from joblib import Parallel, delayed
from tqdm import tqdm
import threading
import logging
import boto3
import os

__all__ = ['download_and_convert']
DATA_FOLDER = os.path.expanduser('~/workspace/invoice/data_exp2')

def _check_duplicate(path):
    return os.path.exists(path)

def _download_and_convert(guid, account, train, thread_storage, image_format):
    if getattr(thread_storage, 's3_client', None) is None:
        thread_storage.s3client = boto3.client('s3', 'us-east-1')
    if getattr(thread_storage, 'logger', None) is None:
        thread_storage.logger = get_logger()
    bucket_name = 'appfolio-ml-invoice-{}-set'.format('training' if train else 'testing')
    fname_prefix = 'attachmentsParallelized/{}/attachments/{}/original'.format(account, guid)
    objects = thread_storage.s3client.list_objects(Bucket=bucket_name, Prefix=fname_prefix).get('Contents', None)
    if objects is None:
        #raise FileNotFoundError('Didn\'t find {} in {}'.format(guid, account))
        thread_storage.logger.error('Didn\'t find {} in {}'.format(guid, account))
        return
    elif len(objects) > 1:
        #raise RuntimeError('Expected one key to match prefix {}, but found {}'.format(fname_prefix, len(objects)))
        thread_storage.logger.error('Expected one key to match prefix {}, but found {}'.format(fname_prefix, len(objects)))
        return
    else:
        key = objects[0]['Key']
    pdf_path = os.path.join(get_dir(os.path.join(DATA_FOLDER, 'pdf', 'train' if train else 'test')), '{}.pdf'.format(guid))
    img_path = os.path.join(get_dir(os.path.join(DATA_FOLDER, 'img', 'train' if train else 'test')), '{}.png'.format(guid))
    if not _check_duplicate(pdf_path):
        thread_storage.s3client.download_file(bucket_name, key, pdf_path)
    gs_device = '{}gray'.format(image_format)
    command = 'gs -q -dNOPAUSE -sDEVICE={} -r300 -dINTERPOLATE -dFirstPage=1 -dLastPage=1 -dGraphicsAlphaBits=4 -g2550x3300 -dPDFFitPage -dUseCropBox\
            -sOutputFile={} -c 30000000 setvmthreshold -f {} -c quit'
    if not _check_duplicate(img_path):
        os.system(command.format(gs_device, img_path, pdf_path))

def download_and_convert(*guids, n_jobs=-1, logger=get_logger(), train=True, image_format='png'):
    thread_storage = threading.local()
    logger.info('Downloading {} invoices in pdfs'.format(len(guids)))
    Parallel(n_jobs=n_jobs, backend='threading', verbose=int(logger.getEffectiveLevel() in [logging.DEBUG, logging.INFO]))\
            (delayed(_download_and_convert)(guid, account, train, thread_storage, image_format)\
            for guid, account in tqdm(zip(guids, get_labels(*guids, account=True, flatten=True)), total=len(guids)))

