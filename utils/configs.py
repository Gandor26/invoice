from .misc import get_dir
import os

DATA_FOLDER = get_dir(os.path.expanduser('~/workspace/invoice/data'))
WAREHOUSE = get_dir(os.path.expanduser('~/workspace/invoice/warehouse'))
IMAGE_FORMAT = 'png'

DB_NAME = 'invoice'
OVERALL_COLLECTION = 'invoice_details'
TRAIN_COLLECTION = 'train_set'
TEST_COLLECTION = 'test_set'
GUID_FIELD_NAME = 'attachment_guid'
VHOST_FIELD_NAME = 'vhost'
ACCOUNT_FIELD_NAME = 'account_name'
VENDOR_FIELD_NAME = 'vendor_id'
VENDOR_NAME_FIELD_NAMES = ['first_name', 'last_name', 'company_name', 'uses_company_name']
PROPERTY_FIELD_NAME = 'property_id'
PROPERTY_ADDRESS_FIELD_NAMES = ['property_name', 'property_address1', 'property_address2', 'property_city', 'property_state', 'property_postal_code']
TOTAL_FIELD_NAME = 'total'
DATASET_FIELD_NAME = 'dataset'

AWS_TRAINING_BUCKET = 'appfolio-ml-invoice-training-set'
AWS_TEST_BUKCET = 'appfolio-ml-invoice-testing-set'

GOOGLE_INPUT_BUCKET = 'af-ml-invoice-training-set'
GOOGLE_OUTPUT_BUCKET = 'xiaoyong-ocr-bucket'
GOOGLE_MIME_TYPE = 'application/pdf'

SCALED_IMAGE_SIZE = 224
