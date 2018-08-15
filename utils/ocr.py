from .database import get_labels
from .misc import get_logger
from joblib import Parallel, delayed
from google.cloud import vision_v1p2beta1 as gv
from google.cloud import storage as gs
import threading
import time
import os

__all__ = ['google_pdf_ocr']

LOCAL_DUMP = os.path.expanduser('~/workspace/invoice/data/ocr')
INPUT_BUCKET = 'af-ml-invoice-training-set'
OUTPUT_BUCKET = 'xiaoyong-ocr-bucket'
MIME_TYPE = 'application/pdf'

def _check_duplicate(guid, storage_client):
    if os.path.exists(os.path.join(LOCAL_DUMP, '{}_output-1-to-1.json'.format(guid))):
        return True
    else:
        dst_bucket = storage_client.get_bucket(bucket_name=OUTPUT_BUCKET)
        blob = list(dst_bucket.list_blobs(prefix=guid))
        if len(blob) < 1:
            return False
        else:
            _download_ocr_file(guid, storage_client)
            return True

def _locate_src_file(guid, storage_client):
    src_bucket = storage_client.get_bucket(bucket_name=INPUT_BUCKET)
    account = get_labels(guid, account=True)[0]
    src_uri_prefix = 'attachmentsParallelized/{}/attachments/{}/original'.format(account, guid)
    blob = list(src_bucket.list_blobs(prefix=src_uri_prefix))[0]
    return 'gs://{}/{}'.format(src_bucket.name, blob.name)

def _locate_dst_file(guid):
    return 'gs://{}/{}_'.format(OUTPUT_BUCKET, guid)

def _download_ocr_file(guid, storage_client):
    dst_bucket = storage_client.get_bucket(bucket_name=OUTPUT_BUCKET)
    blobs = list(dst_bucket.list_blobs(prefix=guid))
    while len(blobs) < 1:
        time.sleep(5)
        blobs = list(dst_bucket.list_blobs(prefix=guid))
    blob = blobs[0]
    blob.download_to_filename(os.path.join(LOCAL_DUMP, blob.name))

def _google_pdf_ocr(guid, thread_storage):
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

    src_uri = _locate_src_file(guid, storage_client)
    gcs_src = gv.types.GcsSource(uri=src_uri)
    input_config = gv.types.InputConfig(gcs_source=gcs_src, mime_type=MIME_TYPE)
    dst_uri = _locate_dst_file(guid)
    gcs_dst = gv.types.GcsDestination(uri=dst_uri)
    output_config = gv.types.OutputConfig(gcs_destination=gcs_dst, batch_size=1)
    feature = gv.types.Feature(type=gv.enums.Feature.Type.DOCUMENT_TEXT_DETECTION)

    async_request = gv.types.AsyncAnnotateFileRequest(features=[feature], input_config=input_config, output_config=output_config)
    vision_client.async_batch_annotate_files(requests=[async_request])
    _download_ocr_file(guid, storage_client)

def google_pdf_ocr(*guids, n_jobs=8):
    thread_storage = threading.local()
    print('Queuing {} invoices on google OCR'.format(len(guids)))
    Parallel(n_jobs=n_jobs, backend='threading', verbose=False)(delayed(_google_pdf_ocr)
            (guid, thread_storage) for guid in guids)
