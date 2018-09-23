# -*- coding: utf-8 -*-
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Penn Tree Bank model')
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data file')
parser.add_argument('--model', type=str, default='small',
                    help='model to run')
parser.add_argument('--data_path', type=str, default='data/',
                    help='location of the data file')
parser.add_argument('--save_path', type=str, default=None,
                    help='Model output directory.')
parser.add_argument('--use_fp16', action='store_true',
                    help='Train using 16-bit floats instead of 32bit floats')

parser.add_argument('--v100', action='store_true',
                    help='use v100 gpu')

parser.add_argument('--verbose', action='store_false',
                    help='with flag to disable more details info')


parser.add_argument('--num_gpus', type=int, default=1,
                    help='If larger than 1, Grappler AutoParallel optimizer')

parser.add_argument('--rnn_mode', type=str, default=None,
                    help='The low level implementation of lstm cell: one of CUDNN')
parser.add_argument('--pngpath', type=str, default='pictures/',
                    help='png path')
parser.add_argument('--hidden_unit', type=int, default=50,
                    help='Hidden unit of the LSTM layer')
parser.add_argument('--max_epoch', type=int, default=4,
                    help='Epoch for decreasing learning rate')
parser.add_argument('--max_max_epoch', type=int, default=20,
                    help='Epoch for running models')
parser.add_argument('--init_scale', type=float, default=0.04,
                    help='init scale')
parser.add_argument('--max_grad_norm', type=int, default=10,
                    help='max_grad_norm')
parser.add_argument('--num_layers', type=int, default=2,
                    help='num of lstm layers')
parser.add_argument('--num_steps', type=int, default=35,
                    help='number of steps of Backpropagation')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--vocab_size', type=int, default=10000,
                    help='vocab_size')

parser.add_argument('--bo_mode', type=str, default="gpyopt",
                    help='Select bo library gpyopt or fmfn(bayeopt)')
parser.add_argument('--bo_epoch', type=int, default=20,
                    help='epochs of BO')
parser.add_argument('--batch_filename', type=str, default='test_test',
                    help='filename of batch file')

parser.add_argument('--tune_parameter', type=str, default='learning_rate',
                    help='learning_rate, dropout_input, dropout_output, dropout_update')
parser.add_argument('--acquisition', type=str, default='EI',
                    help='EI  MPI   LCB  EI_MCMC  LCB_MCMC MPI_MCMC')
parser.add_argument('--kernel', type=str, default='Matern52',
                    help='Select kernels Matern32  Matern52  ExpQuad')
parser.add_argument('--run_single',action='store_true',
                    help='Select single mode, not BO')
parser.add_argument('--fix_lengthscale',action='store_true',
                    help='fix the value of length scale')

parser.add_argument('--decay_expo',action='store_true',
                    help='run lr_decay withexponent 0.95^epoch')

parser.add_argument('--run_single_values',type=float, default=1.0,
                    help='Set the value of current tuning parameter')
parser.add_argument('--length_scale', type=float, default=1.0,
                    help='Select initialisation of length scale')
parser.add_argument('--lr', type=float, default=1.0,
                    help='default learning rate')
parser.add_argument('--lr_decay', type=float, default=1/1.15,
                    help='default learning rate')
parser.add_argument('--k_input', type=float, default=0.5,
                    help='default dropout input 0.99 == donot drop anything, 0.01 == drop all')
parser.add_argument('--k_output', type=float, default=0.5,
                    help='default dropout output')
parser.add_argument('--k_update', type=float, default=0.5,
                    help='default dropout update')

args = parser.parse_args()
args.tied = True





#flags = tf.flags
#
#flags.DEFINE_string(
#    "params", "lr",
#    "A type of model. Possible options are: small, medium, large.")
#
#flags.DEFINE_string(
#    "model", "small",
#    "A type of model. Possible options are: small, medium, large.")
#flags.DEFINE_string("data_path", None,
#                    "Where the training/test data is stored.")
#flags.DEFINE_string("save_path", None,
#                    "Model output directory.")
#flags.DEFINE_bool("use_fp16", False,
#                  "Train using 16-bit floats instead of 32bit floats")
#flags.DEFINE_integer("num_gpus", 1,
#                     "If larger than 1, Grappler AutoParallel optimizer "
#                     "will create multiple training replicas with each GPU "
#                     "running one replica.")
#flags.DEFINE_string("rnn_mode", None,
#                    "The low level implementation of lstm cell: one of CUDNN, "
#                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
#                    "and lstm_block_cell classes.")
#FLAGS = flags.FLAGS


