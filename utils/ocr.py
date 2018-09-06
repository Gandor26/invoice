from .configs import *
from .database import get_labels
from .download import DATA_FOLDER, WAREHOUSE
from .misc import get_logger, get_dir
from joblib import Parallel, delayed
from google.cloud import vision_v1p2beta1 as gv
from google.cloud import storage as gs
from shutil import copy
from tqdm import tqdm
import threading
import time
import os

__all__ = ['google_pdf_ocr']


def _check_duplicate(guid, storage_client):
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
        dst_bucket = storage_client.get_bucket(bucket_name=GOOGLE_OUTPUT_BUCKET)
        blob = list(dst_bucket.list_blobs(prefix=guid))
        return len(blob)>0

def _locate_src_file(guid, account, storage_client):
    src_bucket = storage_client.get_bucket(bucket_name=GOOGLE_INPUT_BUCKET)
    src_uri_prefix = 'attachmentsParallelized/{}/attachments/{}/original'.format(account, guid)
    blob = list(src_bucket.list_blobs(prefix=src_uri_prefix))[0]
    return 'gs://{}/{}'.format(src_bucket.name, blob.name)

def _locate_dst_file(guid):
    return 'gs://{}/{}_'.format(GOOGLE_OUTPUT_BUCKET, guid)

def _google_pdf_ocr(guid, account, thread_storage):
    if getattr(thread_storage, 'vision_client', None) is None:
        thread_storage.vision_client = gv.ImageAnnotatorClient()
    vision_client = thread_storage.vision_client
    if getattr(thread_storage, 'storage_client', None) is None:
        thread_storage.storage_client = gs.Client()
    storage_client = thread_storage.storage_client
    if getattr(thread_storage, 'logger', None) is None:
        thread_storage.logger = get_logger()
    logger = thread_storage.logger
    if _check_duplicate(guid, storage_client):
        logger.warn('Skipping {} because it has already been OCRd'.format(guid))
        return

    src_uri = _locate_src_file(guid, account, storage_client)
    gcs_src = gv.types.GcsSource(uri=src_uri)
    input_config = gv.types.InputConfig(gcs_source=gcs_src, mime_type='application/pdf')
    dst_uri = _locate_dst_file(guid)
    gcs_dst = gv.types.GcsDestination(uri=dst_uri)
    output_config = gv.types.OutputConfig(gcs_destination=gcs_dst, batch_size=1)
    feature = gv.types.Feature(type=gv.enums.Feature.Type.DOCUMENT_TEXT_DETECTION)

    async_request = gv.types.AsyncAnnotateFileRequest(features=[feature], input_config=input_config, output_config=output_config)
    vision_client.async_batch_annotate_files(requests=[async_request])


def google_pdf_ocr(*guids, n_jobs=-1):
    thread_storage = threading.local()
    print('Queuing {} invoices on google OCR'.format(len(guids)))
    accounts = get_labels(*guids, account=True, flatten=True)
    Parallel(n_jobs=n_jobs, backend='threading', verbose=False)(delayed(_google_pdf_ocr)
            (guid, account, thread_storage) for guid, account in tqdm(zip(guids, accounts), total=len(guids)))

