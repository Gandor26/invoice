import argparse
from engine import Engine

def main(*unused_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='DIR', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--mmtm', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--dump_dir', metavar='DIR', type=str, default='./data/dump')

    args = parser.parse_args()
    engine = Engine(args)
    engine.train(args.num_epochs)

if __name__ == '__main__':
    main()
