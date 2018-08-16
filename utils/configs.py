from .misc import get_dir
import os

DATA_FOLDER = get_dir(os.path.expanduser('~/workspace/invoice/data'))
WAREHOUSE = get_dir(os.path.expanduser('~/workspace/invoice/warehouse'))
LOCAL_OCR_DUMP = get_dir(os.path.join(DATA_FOLDER, 'ocr'))
WARE_OCR_DUMP = get_dir(os.path.join(WAREHOUSE, 'ocr'))
IMAGE_FORMAT = 'png'

DB_NAME = 'invoice'
TRAIN_COLLECTION = 'train_set'
TEST_COLLECTION = 'test_set'
GUID_FIELD_NAME = 'attachment_guid'
VHOST_FIELD_NAME = 'vhost'
ACCOUNT_FIELD_NAME = 'account_name'
VENDOR_FIELD_NAME = 'vendor_id'
PROPERTY_FIELD_NAME = 'property_id'
TOTAL_FIELD_NAME = 'total'

GOOGLE_INPUT_BUCKET = 'af-ml-invoice-training-set'
GOOGLE_OUTPUT_BUCKET = 'xiaoyong-ocr-bucket'
GOOGLE_MIME_TYPE = 'application/pdf'
