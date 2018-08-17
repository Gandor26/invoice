from .configs import *
from .database import get_labels
from .misc import get_dir, get_logger
from google.cloud import storage as gs
from joblib import Parallel, delayed
from tqdm import tqdm
from shutil import copy
import threading
import logging
import boto3
import os

__all__ = ['download_and_convert', 'download_ocr']

def _check_duplicate_pdf(guid, train):
    data_path = os.path.join(DATA_FOLDER, 'pdf', 'train' if train else 'test', '{}.pdf'.format(guid))
    ware_path = os.path.join(WAREHOUSE, 'pdf', 'train' if train else 'test', '{}.pdf'.format(guid))
    if os.path.exists(data_path):
        if not os.path.exists(ware_path):
            copy(data_path, ware_path)
        return True
    elif os.path.exists(ware_path):
        copy(ware_path, data_path)
        return True
    else:
        return False

def _check_duplicate_img(guid, train, image_format):
    data_path = os.path.join(DATA_FOLDER, 'img', 'train' if train else 'test', '{}.{}'.format(guid, image_format))
    ware_path = os.path.join(WAREHOUSE, 'img', 'train' if train else 'test', '{}.{}'.format(guid, image_format))
    if os.path.exists(data_path):
        if not os.path.exists(ware_path):
            copy(data_path, ware_path)
        return True
    elif os.path.exists(ware_path):
        copy(ware_path, data_path)
        return True
    else:
        return False

def _check_duplicate_json(guid):
    data_path = os.path.join(DATA_FOLDER, 'ocr', '{}_output-1-to-1.json'.format(guid))
    ware_path = os.path.join(WAREHOUSE, 'ocr', '{}_output-1-to-1.json'.format(guid))
    if os.path.exists(data_path):
        if not os.path.exists(ware_path):
            copy(data_path, ware_path)
        return True
    elif os.path.exists(ware_path):
        copy(ware_path, data_path)
        return True
    else:
        return False

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
    img_path = os.path.join(get_dir(os.path.join(DATA_FOLDER, 'img', 'train' if train else 'test')), '{}.{}'.format(guid, image_format))
    if not _check_duplicate_pdf(guid, train):
        thread_storage.s3client.download_file(bucket_name, key, pdf_path)
        _check_duplicate_pdf(guid, train)
    else:
        thread_storage.logger.warn('PDF of {} already dumped locally'.format(guid))
    gs_device = '{}gray'.format(image_format)
    command = 'gs -q -dNOPAUSE -sDEVICE={} -r300 -dINTERPOLATE -dFirstPage=1 -dLastPage=1 -dGraphicsAlphaBits=4 -dPDFFitPage -dUseCropBox\
            -sOutputFile={} -c 30000000 setvmthreshold -f {} -c quit'
    if not _check_duplicate_img(guid, train, image_format):
        os.system(command.format(gs_device, img_path, pdf_path))
        _check_duplicate_img(guid, train, image_format)
    else:
        thread_storage.logger.warn('Image of {} already dumped locally'.format(guid))

def download_and_convert(*guids, n_jobs=-1, logger=get_logger(), train=True, image_format=IMAGE_FORMAT):
    thread_storage = threading.local()
    logger.info('Downloading {} invoices in pdfs'.format(len(guids)))
    Parallel(n_jobs=n_jobs, backend='threading', verbose=int(logger.getEffectiveLevel() in [logging.DEBUG, logging.INFO]))\
            (delayed(_download_and_convert)(guid, account, train, thread_storage, image_format)\
            for guid, account in tqdm(zip(guids, get_labels(*guids, account=True, flatten=True)), total=len(guids)))

def _download_ocr_file(guid, thread_storage):
    if getattr(thread_storage, 'client', None) is None:
        thread_storage.client = gs.Client()
    client = thread_storage.client
    if getattr(thread_storage, 'logger', None) is None:
        thread_storage.logger = get_logger()
    logger = thread_storage.logger
    dst_bucket = client.get_bucket(bucket_name=GOOGLE_OUTPUT_BUCKET)
    blobs = list(dst_bucket.list_blobs(prefix=guid))
    if len(blobs) < 1:
        logger.error('Cannot find OCR output file of {}'.format(guid))
    else:
        blob = blobs[0]
        ocr_path = os.path.join(get_dir(os.path.join(DATA_FOLDER, 'ocr')), blob.name)
        if not _check_duplicate_json(guid):
            blob.download_to_filename(ocr_path)
            _check_duplicate_json(guid)
        else:
            logger.warn('JSON of {} already dumped locally'.format(guid))

def download_ocr(*guids, n_jobs=-1, logger=get_logger()):
    thread_storage = threading.local()
    logger.info('Downloading {} OCRed invoices'.format(len(guids)))
    Parallel(n_jobs=n_jobs, backend='threading', verbose=int(logger.getEffectiveLevel() in [logging.DEBUG, logging.INFO]))\
            (delayed(_download_ocr_file)(guid, thread_storage) for guid in tqdm(guids))
