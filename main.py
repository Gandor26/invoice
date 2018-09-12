import argparse
from engine import ImageBoWEngine as Engine

def main(*unused_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='DIR', type=str)
    parser.add_argument('--dump_dir', metavar='DIR', type=str, default='./dump')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--margin', type=int)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mmtm', type=float, default=0)
    parser.add_argument('--L1_decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_training_samples', type=int, default=100000)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=1)

    args = parser.parse_args()
    engine = Engine(args)
    engine.train(args.num_epochs)

if __name__ == '__main__':
    main()
