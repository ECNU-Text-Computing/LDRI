import argparse
from email.policy import default

args = argparse.ArgumentParser()

# general
args.add_argument('--batch_size', type=int, default=1024)
args.add_argument('--epochs', type=int, default=48)
args.add_argument('--seed', type=int, default=2024)

args.add_argument('--num_samples', type=int, default=None)  # None, int
args.add_argument('--is_train', type=int, default=0)  # True, False
args.add_argument('--is_valid', type=int, default=1)

args.add_argument('--dataset', type=str,
                  default='kuairand_pure',
                  choices=['kuairand_1k', 'kuairand_pure'])

args.add_argument('--backbone',type=str, default='NFM')

args.add_argument('--n', type=float, default=1)
args.add_argument('--alpha', type=float, default=0.6)
args.add_argument('--beta', type=float, default=0.5)

args.add_argument('--load_epoch', type=int, default=2)
args = args.parse_args()



