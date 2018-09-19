import os

DATA_FOLDER = os.path.expanduser('~/workspace/invoice/data')
WAREHOUSE = os.path.expanduser('~/workspace/invoice/warehouse')
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
VENDOR_ADDRESS_FIELD_NAMES = ['vendor_address1', 'vendor_address2', 'vendor_city', 'vendor_state', 'vendor_postal_code']
PROPERTY_FIELD_NAME = 'property_id'
PROPERTY_ADDRESS_FIELD_NAMES = ['property_name', 'property_address1', 'property_address2', 'property_city', 'property_state', 'property_postal_code']
TOTAL_FIELD_NAME = 'total'
DATASET_FIELD_NAME = 'dataset'

AWS_TRAINING_BUCKET = 'appfolio-ml-invoice-training-set'
AWS_TEST_BUKCET = 'appfolio-ml-invoice-testing-set'

GOOGLE_TRAINING_BUCKET = 'af-ml-invoice-training-set'
GOOGLE_TEST_BUCKET = 'af-ml-invoice-testing-set'
GOOGLE_OCR_BUCKET = 'xiaoyong-ocr-bucket'
GOOGLE_MIME_TYPE = 'application/pdf'

SCALED_IMAGE_SIZE = 224
STOP_WORDS = frozenset(["a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])
