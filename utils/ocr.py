from .configs import *
from .database import get_labels, get_dataset
from .download import DATA_FOLDER, WAREHOUSE
from .misc import get_logger
from joblib import Parallel, delayed
from google.cloud import vision_v1p2beta1 as gv
from google.cloud import storage as gs
from shutil import copy
from tqdm import tqdm
import threading
import os

__all__ = ['google_pdf_ocr']

def _upload_pdf(bucket, guid, account):
    remote_path = 'attachmentsParallelized/{}/attachments/{}/original.pdf'.format(account, guid)
    local_path = os.path.join(DATA_FOLDER, 'pdf', '{}.pdf'.format(guid))
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)

def _check_duplicate(guid, thread_storage):
    storage_client = thread_storage.storage_client
    local_path = os.path.join(DATA_FOLDER, 'ocr', '{}_output-1-to-1.json'.format(guid))
    ware_path = os.path.join(WAREHOUSE, 'ocr', '{}_output-1-to-1.json'.format(guid))
    if os.path.exists(local_path):
        if not os.path.exists(ware_path):
            copy(local_path, ware_path)
        return True
    elif os.path.exists(ware_path):
        copy(ware_path, local_path)
        return True
    else:
        dst_bucket = storage_client.get_bucket(bucket_name=GOOGLE_OCR_BUCKET)
        blob = list(dst_bucket.list_blobs(prefix=guid))
        return len(blob)>0

def _locate_src_file(guid, account, train, thread_storage):
    storage_client = thread_storage.storage_client
    bucket = GOOGLE_TRAINING_BUCKET if train else GOOGLE_TEST_BUCKET
    src_bucket = storage_client.get_bucket(bucket_name=bucket)
    src_uri_prefix = 'attachmentsParallelized/{}/attachments/{}/original'.format(account, guid)
    blobs = list(src_bucket.list_blobs(prefix=src_uri_prefix))
    if len(blobs) < 1:
        thread_storage.logger.warning('The pdf file of {} is not found in Google Cloud Storage, uploading from local dump'.format(guid))
        _upload_pdf(src_bucket, guid, account)
        blobs = list(src_bucket.list_blobs(prefix=src_uri_prefix))
    blob = blobs[0]
    return 'gs://{}/{}'.format(src_bucket.name, blob.name)

def _locate_dst_file(guid):
    return 'gs://{}/{}_'.format(GOOGLE_OCR_BUCKET, guid)

def _google_pdf_ocr(guid, account, train, thread_storage):
    if getattr(thread_storage, 'vision_client', None) is None:
        thread_storage.vision_client = gv.ImageAnnotatorClient()
    if getattr(thread_storage, 'storage_client', None) is None:
        thread_storage.storage_client = gs.Client()
    if getattr(thread_storage, 'logger', None) is None:
        thread_storage.logger = get_logger('utils.ocr')
    if _check_duplicate(guid, thread_storage):
        thread_storage.logger.warning('Skipping {} because it has already been OCRd'.format(guid))
        return

    src_uri = _locate_src_file(guid, account, train, thread_storage)
    gcs_src = gv.types.GcsSource(uri=src_uri)
    input_config = gv.types.InputConfig(gcs_source=gcs_src, mime_type=GOOGLE_MIME_TYPE)
    dst_uri = _locate_dst_file(guid)
    gcs_dst = gv.types.GcsDestination(uri=dst_uri)
    output_config = gv.types.OutputConfig(gcs_destination=gcs_dst, batch_size=1)
    feature = gv.types.Feature(type=gv.enums.Feature.Type.DOCUMENT_TEXT_DETECTION)

    async_request = gv.types.AsyncAnnotateFileRequest(features=[feature], input_config=input_config, output_config=output_config)
    thread_storage.vision_client.async_batch_annotate_files(requests=[async_request])


def google_pdf_ocr(*guids, n_jobs=-1):
    thread_storage = threading.local()
    logger = get_logger('utils.ocr')
    logger.info('Queuing {} invoices on google OCR'.format(len(guids)))
    accounts = get_labels(*guids, account=True, flatten=True)
    is_in_training_set = get_dataset(*guids)
    Parallel(n_jobs=n_jobs, backend='threading', verbose=False)(delayed(_google_pdf_ocr)(guid, account, train, thread_storage)
            for guid, account, train in tqdm(zip(guids, accounts, is_in_training_set), total=len(guids), desc='Running OCR'))

