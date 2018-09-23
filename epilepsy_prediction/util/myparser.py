import numpy as np
#import torch
import argparse

parser = argparse.ArgumentParser(description='Epilepsy Prediction LSTM model')

parser.add_argument('--data', type=str, default='data/',
                    help='location of the csv data')
parser.add_argument('--weights', type=str, default='weights/',
                    help='location of the weights data')
parser.add_argument('--images', type=str, default='pictures/',
                    help='location of the images data')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--hidden_unit', type=int, default=200,
                    help='hidden units of one layer')
parser.add_argument('--epochs', type=int, default=20,
                    help='default epoch')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout 0.99 = drop all )')
parser.add_argument('--dropout_input', type=float, default=0,
                    help='dropout applied to input embedding layers (0 = no dropout)')
parser.add_argument('--dropout_output', type=float, default=0,
                    help='dropout applied to output layers ')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--sop', type=int, default=30,
                    help='sop - seizure occurrence period')
parser.add_argument('--overlap', type=int, default=1,
                    help='overlap')
parser.add_argument('--predstep', type=int, default=0,
                    help='0 # do not change')
parser.add_argument('--l2weight', type=float, default=0.01,
                    help='l2 weight')
parser.add_argument('--seq_length', type=int, default=50,
                    help='sequence length')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer name')
parser.add_argument('--mode', type=str, default='small',
                    help='small medium large models')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--scaler', action='store_false',
                    help='use scaler data')
parser.add_argument('--early_stopping', action='store_false',
                    help='use early stopping')
parser.add_argument('--firing_prob', type=float,default=0.5,
                    help='treshold of the probability at which a \
                    pattern is classified as pre-ictal')
parser.add_argument('--firing_threshold',  type=float,default=0.5,
                    help='threshold for alarm generation using the firing power filter')
parser.add_argument('--firing_filter_window', type=int,default=20,
                    help='size of the filter window')
parser.add_argument('--img_path', type=str,default='0',
                    help='path of saving plot images')
parser.add_argument('--result_filename', type=str,default='default.txt',
                    help='result save path')
parser.add_argument('--weight_num', type=str,default='0',
                    help='model weight num (use which model?)')
parser.add_argument('--activation', type=str,default='softmax',
                    help='activation function of neuron')

parser.add_argument('--batch_filename', type=str, default='none',
                    help='batch_filename')

parser.add_argument('--bo_mode', type=str, default="gpyopt",
                    help='Select bo library gpyopt or fmfn(bayeopt)')
parser.add_argument('--bo_epoch', type=int, default=20,
                    help='epochs of BO')
parser.add_argument('--tune_parameter', type=str, default='learning_rate',
                    help='learning_rate, dropout_input, dropout_output, dropout_update')



args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility, but this does not work with 
# keras
np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#print(args.cuda)
#if torch.cuda.is_available():
#    if not args.cuda:
#        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#    else:
#        torch.cuda.manual_seed(args.seed)

if args.mode == 'test':
    print('test mode')
    args.lr = 3
    args.epochs = 10
    args.hidden_unit = 1
    args.dropout = 0.99
    

