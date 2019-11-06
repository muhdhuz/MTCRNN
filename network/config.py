"""
Training Options
"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--hidden_size', type=int, default=60,
                    help='no. of hidden nodes for each GRU layer')
#parser.add_argument('--output_size', type=int, default=256,
#                    help='mu-law encode factor = one-hot size = final network layer size')
parser.add_argument('--n_layers', type=int, default=3, help='no of stacked GRU layers')

parser.add_argument('--sample_rate', type=int, default=16000, help='sampling rate for input sound')
parser.add_argument('--seq_len', type=int, default=4000, help='sequence length of each input data in no. of samples')
parser.add_argument('--stride', type=int, default=1, help='shift in no. of samples between adjacent data sequences')
parser.add_argument('--mulaw_channels', type=int, default=256, help='mu-law encoding channels')
parser.add_argument('--batch_size', type=int, default=12, help='minibatch size for training input')

parser.add_argument('--param_dir', type=str, default=None, help='parameter file directory ')
parser.add_argument('--prop', type=str, default=None, nargs='+', help='parameters to be used')
parser.add_argument('--cond_size', type=int, default=0,
                    help='conditional vectors size')


def parse_args(is_training=True):
    if is_training:
        parser.add_argument('--data_dir', type=str, default='./data/audio', help='training data directory')
        parser.add_argument('--output_dir', type=str, default='./output', help='Output dir for saving model and etc')
        parser.add_argument('--num_steps', type=int, default=1, help='Total training steps')
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        parser.add_argument('--checkpoint', type=float, default=11, help='save model every checkpoint steps')
    else:
        parser.add_argument('--model_dir', type=str, required=True, help='Pre-trained model dir')
        parser.add_argument('--step', type=int, default=0, help='A specific step of pre-trained model to use')
        parser.add_argument('--seed', type=str, help='A seed file to generate sound')
        parser.add_argument('--out', type=str, help='Output file name which is generated')

    return parser.parse_args()


def print_help():
    parser.print_help()
