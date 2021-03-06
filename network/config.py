"""
Training Options
"""
import argparse

parser = argparse.ArgumentParser()

#network arguments
parser.add_argument('--hidden_size', type=int, default=800, help='no. of hidden nodes for each GRU layer')
parser.add_argument('--n_layers', type=int, default=3, help='no of stacked GRU layers')
parser.add_argument('--plstm', action='store_true', help='use alternative phasedLSTM architecture instead of GRU. Currently only implemented with net 0')
parser.add_argument('--net', type=int, default=0, choices=range(0, 7),
                    help='architecture for param model. 0=baseline, 1=separate mu/sigma, 2=separate mu/sigma with common GRU layer,\
                         3=separate pathways for each output, 4=separate pathways for each output with common GRU layer,\
                        5=mixture density model, 6 onwards=legacy options to be compatible with older models')

#training/audio data arguments
parser.add_argument('--sample_rate', type=int, default=16000, help='sampling rate for input variable')
parser.add_argument('--seq_len', type=int, default=4000, help='sequence length of each input data in no. of samples')
parser.add_argument('--stride', type=int, default=1, help='shift in no. of samples between adjacent data sequences')
parser.add_argument('--mulaw_channels', type=int, default=256, help='mu-law encoding channels')
parser.add_argument('--batch_size', type=int, default=16, help='minibatch size for training input')

#data arguments
parser.add_argument('--param_dir', type=str, default=None, help='parameter file directory')
parser.add_argument('--prop', type=str, default=[], nargs='+', help='parameters to be used as conditioning')
parser.add_argument('--cond_size', type=int, default=16, help='input vector size: conditional vector')
parser.add_argument('--generate', type=str, default=['audio'], nargs='+', help='parameters/audio to be generated, defaults audio')
parser.add_argument('--gen_size', type=int, default=1, help='input vector size: generated features, if audio only = 1 or one-hot channels')
parser.add_argument('--onehot', action='store_true', help='whether to transform mulaw to onehot prior to input')
parser.add_argument('--temp', type=float, default=1.0, help='temperature param for sampling randomness. Normally 1.0 for training')
#parser.add_argument('--no_shuffle', action='store_false', help='whether to shuffle (randomize) data loading (default:shuffle)')

def parse_args(is_training=True):
    if is_training:
        parser.add_argument('--data_dir', type=str, default='./data/audio', help='training data directory')
        parser.add_argument('--output_dir', type=str, default='./output', help='output dir for saving model and etc')
        parser.add_argument('--num_steps', type=int, default=100000, help='total training steps')
        parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing rate ratio. 1.0=100% teacher forcing')
        parser.add_argument('--ss', type=float, default=None, help='if using scheduled sampling provide an end point. start point=tfr')
        parser.add_argument('--checkpoint', type=float, default=10000, help='save model every checkpoint steps')
        parser.add_argument('--model_dir', type=str, default=None, help='to resume from checkpoint, supply a model dir')
        parser.add_argument('--step', type=int, default=0, help='a specific step of model checkpoint to resume from')
        parser.add_argument('--loss_cutoff', type=float, default=None, help='training will stop if loss drops below this cutoff')
    else: #for generation
        parser.add_argument('--model_dir', type=str, required=True, help='pre-trained model dir')
        parser.add_argument('--step', type=int, default=0, help='a specific step of pre-trained model to use')
        parser.add_argument('--length', type=int, default=16000, help='length of synthesized output in samples')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--seed', type=str, default=None, help='a seed file to load data for priming/self conditioning. Data will be pulled at beginning of file unless seed_start specified')
        group.add_argument('--data_dir', type=str, default=None, help='a test data directory to load data for priming/self conditioning. Data will be pulled randomly')
        parser.add_argument('--seed_start', type=int, default=0, help='starting sample value to load data if seed is specified')
        parser.add_argument('--paramvect',default='self', const='self', nargs='?',choices=('self', 'external','none'),
                    help='source of paramvect. self(default): taken from data file, external: taken from numpy array, none: no conditioning')
        parser.add_argument('--out', type=str, default='generated', help='output file path for generated sequence')
        parser.add_argument('--external_array', type=str, default=None, help='a saved numpy array of shape [batch,length,features] for external conditioning')        
        parser.add_argument('--external_sr', type=int, default=None, help='original sample rate of external conditioning array')
        parser.add_argument('--save', action='store_true', help='save the output (if audio this is automatic)')
        parser.add_argument('--rand_prime', action='store_true', help='overwrites the use of random real data (also external_prime) to use a randomly generated primer. Primer length will be (seq_len-length)')
        parser.add_argument('--rand_seed', type=int, default=None, help='provide a seed for random generation if want to get same number consistently')    
    
    return parser.parse_args()


def print_help():
    parser.print_help()
