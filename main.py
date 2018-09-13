import os
import argparse
from engine import ImageBoWEngine as Engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='DIR', type=str)
    parser.add_argument('--dump_dir', metavar='DIR', type=str, default=os.path.expanduser('~/workspace/invoice/dump'),
            help='the path to the folder where models are dumped')
    parser.add_argument('--tag', type=str, help='a tag to the current model, usually the specified configurations; the tags are used to dump or reload models')
    parser.add_argument('--cuda', action='store_true', help='if enabled, use GPU to do training and evaluation')
    parser.add_argument('--eval', action='store_true', help='if enabled, skip training and only do evaluation with dumped models')
    parser.add_argument('--resume', action='store_true', help='if enabled, resume training based on a partially trained model; usually for fine-tuning')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducing results')
    parser.add_argument('--margin', type=int, default=10, help='the maximum space of margin in data augmentation (cropping or padding)')
    parser.add_argument('--threshold', type=float, default=0, help='if set positive, only classes with at least so many samples are used')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--mmtm', type=float, default=0, help='momentum in SGD optimizer')
    parser.add_argument('--L1_decay', type=float, default=0, help='the weight of L1 regularizer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
    parser.add_argument('--num_training_samples', type=int, default=100000, help='the number of training samples; you can use arbitrarily many training samples by\
            expoliting training set with data augmentation')
    parser.add_argument('--num_epochs', type=int, default=20, help='the number of iterations over training set')
    parser.add_argument('--top_k', type=int, default=1, help='take the most k probable predictions to match groundtruth in evaluation; namely precision@k')

    args = parser.parse_args()
    engine = Engine(args)

    if not args.eval:
        try:
            engine.train(args.num_epochs, args.resume)
        except KeyboardInterrupt:
            pass
    engine.test()



if __name__ == '__main__':
    main()
