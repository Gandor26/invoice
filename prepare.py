#! /usr/local/bin/python
from utils import DATA_FOLDER
from utils import get_top_vhosts, get_guids_by_vhost, get_labels
from utils import download_and_convert, download_ocr, clear_log_file
from collections import defaultdict
from numpy import random
from glob import glob
import argparse
import json
import os

def split_dataset(*vhosts, test_size, seed):
    random.seed(seed)
    guids = []
    guids.extend(get_guids_by_vhost(*vhosts, train=True))
    guids.extend(get_guids_by_vhost(*vhosts, train=False))
    labels = ['{}_{}'.format(*t) for t in get_labels(*guids, vendor=True, train=None, flatten=True)]
    stats = defaultdict(list)
    for guid, label in zip(guids, labels):
        stats[label].append(guid)
    print(len(guids))
    train_set = defaultdict(list)
    test_set = defaultdict(list)
    for label in stats:
        if len(stats[label]) < 2:
            continue
        num_sample = len(stats[label])
        num_test = max(int(num_sample*test_size), 1)
        idx_test = random.choice(num_sample, num_test, replace=False)
        for i in range(num_sample):
            if i in idx_test:
                test_set[label].append(stats[label][i])
            else:
                train_set[label].append(stats[label][i])
    with open(os.path.join(DATA_FOLDER, 'set', 'train.json'), 'w') as f:
        json.dump(train_set, f)
    with open(os.path.join(DATA_FOLDER, 'set', 'test.json'), 'w') as f:
        json.dump(test_set, f)
    return train_set, test_set

def _prepare_invoices(data_split):
    guids = sum(data_split.values(), [])
    download_ocr(*guids)
    files = [os.path.split(f)[-1].split('_')[0] for f in glob(os.path.join(DATA_FOLDER, 'ocr', '*.json'))]
    ocred = [g for g in guids if g in files]
    download_and_convert(*ocred)
    not_ocred = list(set(guid) - set(ocred))
    not_ocred_labels = ['{}_{}'.format(*l) for l in get_labels(*not_ocred, vendor=True, train=None, flatten=True)]
    for guid, label in zip(not_ocred, not_ocred_labels):
        data_split[label].remove(guid)
    return data_split

def prepare_train_set(train_set=None):
    if train_set is None:
        try:
            with open(os.path.join(DATA_FOLDER, 'set', 'train.json'), 'r') as f:
                train_set = json.load(f)
        except FileNotFoundError:
            print('No train split is created, quit..')
            return
    train_set = _prepare_invoices(train_set)
    with open(os.path.join(DATA_FOLDER, 'set', 'train.json'), 'w') as f:
        json.dump(train_set, f)
    print('train_set is good to go')
    return train_set

def prepare_test_set(test_set=None):
    if test_set is None:
        try:
            with open(os.path.join(DATA_FOLDER, 'set', 'test.json'), 'r') as f:
                test_set = json.load(f)
        except FileNotFoundError:
            print('No test split is created, quit..')
            return
    test_set = _prepare_invoices(test_set)
    with open(os.path.join(DATA_FOLDER, 'set', 'test.json'), 'w') as f:
        json.dump(test_set, f)
    print('test_set is good to go')
    return test_set

def create_dataset_folder(train_set=None, test_set=None, image_size=None):
    if train_set is None:
        try:
            with open(os.path.join(DATA_FOLDER, 'set', 'train.json'), 'r') as f:
                train_set = json.load(f)
        except FileNotFoundError:
            print('No train split is created, quit..')
            return
    if test_set is None:
        try:
            with open(os.path.join(DATA_FOLDER, 'set', 'test.json'), 'r') as f:
                test_set = json.load(f)
        except FileNotFoundError:
            print('No test split is created, quit..')
            return
    guids_and_labels_train = [(g, l) for g in train_set[l] for l in train_set]
    guids_and_labels_test = [(g, l) for g in test_set[l] for l in test_set]
    build_dataset(*guids_and_labels_train, train=True, image_size=image_size)
    build_dataset(*guids_and_labels_test, train=False, image_size=image_size)

if __name__ == '__main__':
    clear_log_file()
    parser = argparse.ArgumentParser()
    parser.add_argument('vhosts', nargs='+')
    parser.add_argument('--split', action='store_true',
            help='Performs dataset split. If not specified, make sure you already have a split dump')
    parser.add_argument('--download', action='store_true',
            help='Download invoices and ocr files based on dataset split')
    parser.add_argument('--create', action='store_true',
            help='Create dataset folder based on dataset split')
    parser.add_argument('--seed', type=int, default=42,
            help='Random seed for dataset splitting')
    parser.add_argument('--image_size', type=int,
            help='Dimension of scaled low-resolution images for training')
    parser.add_argument('--test_size', type=float, default=0.1,
            help='Fraction of samples used in test')
    args = parser.parse_args()
    train_set, test_set = None, None
    if args.split:
        train_set, test_set = split_dataset(*args.vhosts, test_size=args.test_size, seed=args.seed)
    if args.download:
        train_set = prepare_train_set(train_set)
        test_set = prepare_test_set(test_set)
    if args.create:
        create_dataset_folder(train_set, test_set)
