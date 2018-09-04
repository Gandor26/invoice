from utils import get_top_vhosts, get_guids_by_vhost, get_labels, build_dataset, clear_log_file
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from collections import Counter
import argparse

def make_dataset(*vhosts, seed, size=None):
    guids = []
    guids.extend(get_guids_by_vhost(*vhosts, train=True))
    guids.extend(get_guids_by_vhost(*vhosts, train=False))
    labels = ['{}_{}'.format(*t) for t in get_labels(*guids, vendor=True, train=None, flatten=True)]
    cnter = Counter(labels)
    guids, labels = zip(*filter(lambda guid_and_label: cnter[guid_and_label[1]]>1, zip(guids, labels)))
    sampler = SSS(n_splits=1, test_size=0.2, random_state=seed)
    idx_train, idx_test = next(sampler.split(guids, labels))
    guids_and_labels_train = [(guids[i], labels[i]) for i in idx_train]
    guids_and_labels_test = [(guids[i], labels[i]) for i in idx_test]
    build_dataset(*guids_and_labels_train, train=True, image_size=size)
    build_dataset(*guids_and_labels_test, train=False, image_size=size)

if __name__ == '__main__':
    clear_log_file()
    parser = argparse.ArgumentParser()
    parser.add_argument('vhosts', nargs='+')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int)
    args = parser.parse_args()
    make_dataset(*args.vhosts, seed=args.seed, size=args.size)
