from .configs import *
from .database import get_labels, get_dataset
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

def _check_duplicate_pdf(guid):
    data_path = os.path.join(DATA_FOLDER, 'pdf', '{}.pdf'.format(guid))
    ware_path = os.path.join(WAREHOUSE, 'pdf', '{}.pdf'.format(guid))
    if os.path.exists(data_path):
        if not os.path.exists(ware_path):
            copy(data_path, ware_path)
        return True
    elif os.path.exists(ware_path):
        copy(ware_path, data_path)
        return True
    else:
        return False

def _check_duplicate_img(guid, image_format):
    data_path = os.path.join(DATA_FOLDER, 'img', '{}.{}'.format(guid, image_format))
    ware_path = os.path.join(WAREHOUSE, 'img', '{}.{}'.format(guid, image_format))
    if os.path.exists(data_path):
        if not os.path.exists(ware_path):
            copy(data_path, ware_path)
        return True
    elif os.path.exists(ware_path):
        copy(ware_path, data_path)
        return True
    else:
        return

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
    if getattr(thread_storage, 'logger', None) is None:
        thread_storage.logger = get_logger('utils.download')
    pdf_path = os.path.join(DATA_FOLDER, 'pdf', '{}.pdf'.format(guid))
    img_path = os.path.join(DATA_FOLDER, 'img', '{}.{}'.format(guid, image_format))
    if _check_duplicate_pdf(guid):
        thread_storage.logger.warn('PDF of {} already dumped locally'.format(guid))
    else:
        if getattr(thread_storage, 's3_client', None) is None:
            thread_storage.s3client = boto3.client('s3', 'us-east-1')
        bucket_name = AWS_TRAINING_BUCKET if train else AWS_TEST_BUKCET
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
        thread_storage.s3client.download_file(bucket_name, key, pdf_path)
        _check_duplicate_pdf(guid)
    gs_device = '{}gray'.format(image_format)
    command = 'gs -q -dNOPAUSE -sDEVICE={} -r300 -dINTERPOLATE -dFirstPage=1 -dLastPage=1 -dGraphicsAlphaBits=4 -dPDFFitPage -dUseCropBox\
            -sOutputFile={} -c 30000000 setvmthreshold -f {} -c quit'
    if  _check_duplicate_img(guid, image_format):
        thread_storage.logger.warn('Image of {} already dumped locally'.format(guid))
    else:
        os.system(command.format(gs_device, img_path, pdf_path))
        _check_duplicate_img(guid, image_format)

def download_and_convert(*guids, n_jobs=-1, image_format=IMAGE_FORMAT):
    thread_storage = threading.local()
    logger = get_logger('utils.download')
    logger.info('Downloading {} invoices in pdfs'.format(len(guids)))
    accounts = get_labels(*guids, account=True, train=None, flatten=True)
    is_in_training_set = get_dataset(*guids)
    Parallel(n_jobs=n_jobs, backend='threading', verbose=False)(delayed(_download_and_convert)(guid, account, train, thread_storage, image_format)
            for guid, account, train in tqdm(zip(guids, accounts, is_in_training_set), total=len(guids), desc='Downloading invoice'))

def _download_ocr_file(guid, thread_storage):
    if getattr(thread_storage, 'logger', None) is None:
        thread_storage.logger = get_logger('utils.download')
    logger = thread_storage.logger
    if _check_duplicate_json(guid):
        logger.warn('JSON of {} already dumped locally'.format(guid))
    else:
        if getattr(thread_storage, 'client', None) is None:
            thread_storage.client = gs.Client()
        client = thread_storage.client
        dst_bucket = client.get_bucket(bucket_name=GOOGLE_OCR_BUCKET)
        blobs = list(dst_bucket.list_blobs(prefix=guid))
        if len(blobs) < 1:
            logger.error('Cannot find OCR output file of {}'.format(guid))
        else:
            blob = blobs[0]
            ocr_path = os.path.join(DATA_FOLDER, 'ocr', blob.name)
            blob.download_to_filename(ocr_path)
            _check_duplicate_json(guid)

def download_ocr(*guids, n_jobs=-1):
    thread_storage = threading.local()
    logger = get_logger('utils.download')
    logger.info('Downloading {} OCRed invoices'.format(len(guids)))
    Parallel(n_jobs=n_jobs, backend='threading', verbose=False)(delayed(_download_ocr_file)(guid, thread_storage)
            for guid in tqdm(guids, desc='Downloading OCR'))
