"""
Training Options
"""
import argparse

parser = argparse.ArgumentParser()

#network arguments
parser.add_argument('--hidden_size', type=int, default=800, help='no. of hidden nodes for each GRU layer')
parser.add_argument('--n_layers', type=int, default=3, help='no of stacked GRU layers')

#training arguments
parser.add_argument('--sample_rate', type=int, default=16000, help='sampling rate for input sound')
parser.add_argument('--seq_len', type=int, default=4000, help='sequence length of each input data in no. of samples')
parser.add_argument('--stride', type=int, default=1, help='shift in no. of samples between adjacent data sequences')
parser.add_argument('--mulaw_channels', type=int, default=256, help='mu-law encoding channels')
parser.add_argument('--batch_size', type=int, default=16, help='minibatch size for training input')

#data arguments
parser.add_argument('--param_dir', type=str, default=None, help='parameter file directory ')
parser.add_argument('--prop', type=str, default=[], nargs='+', help='parameters to be used as conditioning')
parser.add_argument('--cond_size', type=int, default=16, help='input vector size: conditional vector')
parser.add_argument('--generate', type=str, default=['audio'], nargs='+', help='parameters/audio to be generated, defaults audio')
parser.add_argument('--gen_size', type=int, default=1, help='input vector size: generated features, if audio only = 1 or one-hot channels')
#parser.add_argument('--paramonly', action='store_true', help='whether training only on parameters (no audio)')
parser.add_argument('--onehot', action='store_true', help='whether to transform mulaw to onehot prior to input')
parser.add_argument('--temp', type=float, default=0.9, help='temperature param for sampling')

def parse_args(is_training=True):
    if is_training:
        parser.add_argument('--data_dir', type=str, default='./data/audio', help='training data directory')
        parser.add_argument('--output_dir', type=str, default='./output', help='output dir for saving model and etc')
        parser.add_argument('--num_steps', type=int, default=100000, help='total training steps')
        parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing rate')
        parser.add_argument('--checkpoint', type=float, default=10000, help='save model every checkpoint steps')
        parser.add_argument('--model_dir', type=str, default=None, help='to resume from checkpoint, supply a model dir')
        parser.add_argument('--step', type=int, default=0, help='a specific step of model checkpoint to resume from')
    else:
        parser.add_argument('--model_dir', type=str, required=True, help='pre-trained model dir')
        parser.add_argument('--step', type=int, default=0, help='a specific step of pre-trained model to use')
        parser.add_argument('--length', type=int, default=16000, help='length of synthesized output in samples')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--seed', type=str, default=None, help='a seed file to generate sound')
        group.add_argument('--data_dir', type=str, default='./data/audio', help='a test data directory to generate sound')
        parser.add_argument('--paramvect',default='self', const='self', nargs='?',choices=('self', 'external','none'),
                    help='source of paramvect. self(default): taken from data file, external: taken from numpy array, none: no conditioning')
        parser.add_argument('--out', type=str, default='generated', help='output file name which is generated')
        parser.add_argument('--external_array', type=str, default=None, help='a saved numpy array of shape [batch,length,features] for external conditioning')        
        parser.add_argument('--external_sr', type=int, default=None, help='original sample rate of external conditioning array')
        parser.add_argument('--save', action='store_true', help='save the output (if audio this is automatic)')  
    
    return parser.parse_args()


def print_help():
    parser.print_help()
