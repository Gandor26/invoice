from utils import DATA_FOLDER, STOP_WORDS
from utils import get_vendor_name, get_vendor_address
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from glob import glob
import pickle as pk
import json
import os

SKIP_CHAR = ' ,.;:\'\"()[]{}-_+=`~*?!'

def word_filter(s):
    if s in STOP_WORDS:
        return False
    if len(s) == 0:
        return False
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def get_common_words_by_vendor(write=False):
    train_folder = os.path.join(DATA_FOLDER, 'set', 'train')
    vocab = set()
    guids = []
    for cls_folder in glob(os.path.join(train_folder, '*')):
#         if os.path.exists(os.path.join(cls_folder, 'common_words.json')) and not write:
#             with open(os.path.join(cls_folder, 'common_words.json'), 'r') as f:
#                 common_words = json.load(f)
#         else:
#             cnter = Counter()
        files = glob(os.path.join(cls_folder, '*.pkl'))
#             for box_file in files:
#                 with open(box_file, 'rb') as f:
#                     box = pk.load(f)
#                 words = list(filter(word_filter, [s.strip(SKIP_CHAR) for s in box.text.lower().split()]))
#                 cnter.update(words)
#             common_words, _ = zip(*filter(lambda t:t[1]>int(len(files)>1), cnter.most_common(100)))
#             with open(os.path.join(cls_folder, 'common_words.json'), 'w') as f:
#                 json.dump(common_words, f)
#         vocab.update(common_words)
        guids.append(os.path.split(files[0])[-1].split('.')[0])
    vendor_names = get_vendor_name(*guids)
    vendor_name_words = list(filter(word_filter, [s.strip(SKIP_CHAR) for name in vendor_names for s in name.lower().split()]))
    vendor_addresses = get_vendor_address(*guids)
    vendor_addr_words = list(filter(word_filter, [s.strip(SKIP_CHAR) for addr in vendor_addresses for s in addr.lower().split()]))
    vocab.update(vendor_name_words)
    vocab.update(vendor_addr_words)
    return vocab

class BoW(object):
    def __init__(self, vocab=None):
        vocab = vocab or get_common_words_by_vendor()
        self.vectorizer = CountVectorizer(vocabulary=vocab)

    def __call__(self, box):
        words = ' '.join(filter(word_filter, [s.strip(SKIP_CHAR) for s in box.text.lower().split()]))
        vector = self.vectorizer.transform([words]).toarray()[0]
        return vector

