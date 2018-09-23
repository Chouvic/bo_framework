#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 16:17:30 2018

@author: chouvic rodrigo
"""
from __future__ import print_function
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
import numpy as np
import GPyOpt
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adagrad, Adam
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from random import sample, seed
from copy import deepcopy
from sklearn import preprocessing
from sklearn.utils import shuffle
from utils.plot import plot_lists, plot_single_list, plot_single_list_same, plot_scatter_xy
from utils.savefiles import save_str_app, saveResultAppend
from utils.lstmconfig import get_param_domain
import pandas as pd
import os.path
import sys
import datetime

import utils.lstmflags as flag
FLAGS = flag.args
#print(FLAGS)

currentDT = datetime.datetime.now()
data_path = FLAGS.data
images_path = FLAGS.images


overlap = FLAGS.overlap
predstep = FLAGS.predstep
lr = FLAGS.lr
d = FLAGS.dropout
sop = FLAGS.sop
seq_length = FLAGS.seq_length
weight_num = FLAGS.weight_num

train_loss_list = []
validation_loss_list = []
params_list = []

# create sequence data for x and y
# default settings for seq_length is 50, overlap=1, predstep = 0
# x will be transformed from (9336, 24) to (9286, 50, 24)
# y will be transformed from (9336, 2) to (9286, 2)
def create_seq(X_seq,y_seq,seq_length,overlap,predstep):
    nb_samples = X_seq.shape[0]
    new_size = (nb_samples - nb_samples%overlap)/overlap-seq_length-predstep
    num = int(new_size)
    X_0 = np.zeros((num, seq_length, X_seq.shape[1]))
    y_0 = np.zeros((num, y_seq.shape[1]))

    for i in range(0, num):
        j = i * overlap
        X_0[i,:,:] = X_seq[j:j+seq_length,:]
        y_0[i,:] = y_seq[j+seq_length-1,:]
    return X_0, y_0


# build model, current model only has lstm neural network settings
#

def build_net(structure,X_train,y_train,X_test,y_test, \
              batch_size, lr, dropouts, np_epoch, l2weight,optimizer_str):

  print('Build model...')
  print(structure)
  print('learning rate = %f'%(lr))
  model = Sequential()
  n = len(structure)
  im = 2
  for i in range(len(structure)):
    type = structure[i][0]

    if type == 'lstm':
      n_neuron = structure[i][1]
      if i==0 and n == 1:
          model.add(LSTM(n_neuron, input_shape=(seq_length, X_train.shape[2]),
                       return_sequences=False, dropout=dropouts,implementation=im, recurrent_dropout=dropouts,
                       kernel_regularizer=l2(l2weight),recurrent_regularizer=l2(l2weight)))
      elif i==0 and n != 1:
          model.add(LSTM(n_neuron, input_shape=(seq_length, X_train.shape[2]),
                         return_sequences=True, dropout=dropouts,implementation=im, recurrent_dropout=dropouts,
                       kernel_regularizer=l2(l2weight),recurrent_regularizer=l2(l2weight)))
      elif i>0 and i == n-1:
          model.add(LSTM(n_neuron, return_sequences=False, dropout=dropouts,implementation=im, recurrent_dropout=dropouts,
                       kernel_regularizer=l2(l2weight),recurrent_regularizer=l2(l2weight)))
      else:
          model.add(LSTM(n_neuron, return_sequences=True, dropout=dropouts, implementation=im,recurrent_dropout=dropouts,
                       kernel_regularizer=l2(l2weight),recurrent_regularizer=l2(l2weight)))

  model.add(Dense(y_train.shape[1], activity_regularizer = l2(0.0)))
  model.add(Activation(FLAGS.activation))

  if optimizer_str == 'adam':
    optimizer = Adam(lr=lr)
  elif optimizer_str == 'RMSprop':
    optimizer = RMSprop(lr=lr)

  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  # checkpointer = ModelCheckpoint(filepath="+PredictionAlgorithms/+Artificial_Neural_Networks/+Types/tmp/weights1.hdf5", verbose=1, save_best_only=True)
  checkpointer = ModelCheckpoint(filepath="weights/weights"+ weight_num+".hdf5", verbose=0, save_best_only=True)

  if not FLAGS.early_stopping:
      print('no early stopping')
      callbacks = [History(),checkpointer]
  else:
      earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto')
      callbacks = [History(),checkpointer,earlystopping]

  print('n_int = %d n_preict = %d train_ratio= %f' % (n_int, n_preict, n_preict/n_int))
  print('n_valid_int= %d n_valid_predict=%d valid_ratio = %f' % (n_int_valid, n_preict_valid, n_preict_valid/n_int_valid))

  if y_train.shape[1] == 2:
    class_weights = {0:1, 1:1.*n_int/n_preict}
  else:
    class_weights = {0:1, 1:1.*n_int/n_preict, 2:1.*n_int/n_ict,3:1.*n_int/n_posict }

  print(str(class_weights))

  hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=np_epoch,
                    validation_data=(X_test,y_test),class_weight = class_weights,callbacks=callbacks,verbose=2)

  # save train and validation loss results to files
  config = FLAGS.tune_parameter + "_" +str(FLAGS.lr)
  savelossfilename = FLAGS.batch_filename+"_"+"train_validation_losses.txt"
  saveResultAppend(savelossfilename, 'Configuration', config)
  saveResultAppend(savelossfilename, 'train_loss', hist.history['loss'])
  saveResultAppend(savelossfilename, 'val_loss', hist.history['val_loss'])

#  plot_lists(hist.history['loss'], hist.history['val_loss'], \
#             filename=images_path+'loss_val'+FLAGS.img_path, \
#             l1name='train_loss', l2name='validation_loss',\
#           xlabel='epochs', ylabel='loss', title='train and validation loss')
#
#
#  plot_lists(hist.history['acc'], hist.history['val_acc'], \
#             filename=images_path+'acc_valacc'+FLAGS.img_path, \
#             l1name='train_acc', l2name='validation_acc',\
#           xlabel='epochs', ylabel='accuracy', title='train and validation accuracy')

  model.load_weights("weights/weights"+ weight_num+".hdf5")

  return model

def idx_seizure(y):
  """
  Inputs
    y - target matrix
  Outputs
    idx - indexes of where each seizure begins and ends
  """
  idx = []
  print(y.shape)
  print(y.shape[0])

  for i in range(0,y.shape[0]):
    if y[i,1] == 1 and y[i-1,1] == 0 or (i==0 and y[0,1] == 1):
      for j in range(i, int(i+sop*3*60/5+1)):
        try:
          if y[j,1] == 1 and y[j+1,1] == 0:
            idx.append([i,j])
            break
        except IndexError:
          idx.append([i,j])
          break
  return idx

def firing_power(pred,y,th,prob,n):
  """
  Inputs
    pred - predictions matrix
    y - target matrix
    th - threshold for alarm generation using the firing power filter
    n - size of the filter window
    prob - treshold of the probability at which a pattern is classified
           as pre-ictal
  Outputs
    new_pred - prediction matrix after applying the filter
    new_y - target matrix
  """
  size = pred.shape[0]
  new_pred = np.zeros((size-n,2))
  new_y = deepcopy(y)
  new_y = new_y[n:,:]
  for i in range(n,size):
    # fp = pred[pred[i-n:i,1]>=prob].shape[0]/float(n)
    fp = np.sum(pred[i-n:i,1]>=prob)/float(n)
    if fp >= th:
      new_pred[i-n,1] = 1
    else:
      new_pred[i-n,0] = 1
  for i in range(new_pred.shape[0]):
    if new_pred[i,1] == 1:
      for j in range(1,int(sop*60/5+1)):
        try:
          new_pred[i+j,1] = 0
          new_pred[i+j,0] = 1
        except IndexError:
          pass
  return new_pred, new_y

def performance(pred,y,idx):
  """
  Inputs
   pred - prediction matrix, after firing power application
   y - target matrix
   idx - indexes of where each seizure begins and ends
  Outputs
    sens - sensitivity
    fpr - false positive rate
  """
  fp = 0.
  tp = 0.
  for i in range(pred.shape[0]):
    if np.argmax(y[i,:]) == 0 and np.argmax(pred[i,:]) == 1:
      fp += 1
  FPR = fp/(y[y[:,0]==1].shape[0]-fp*sop*60/5)*720
  for i in idx:
    for j in range(i[0],i[1]+1):
      try:
        if np.argmax(pred[j,:]) == 1:
          tp += 1
          break
      except IndexError:
          break
  sens = float(tp)/len(idx)
  print("true positive = "+str(tp))
  print("total positive = "+str(len(idx)))
  return FPR, sens

front_str = "toy2_"
#front_str = ""

# Read csv files to train and validation variables
X_train_o = pd.read_csv(data_path+front_str+'xtrain.csv',header=None)
y_train_o = pd.read_csv(data_path+front_str+'ytrain.csv',header=None)
X_valid_o = pd.read_csv(data_path+front_str+'xvalid.csv',header=None)
y_valid_o = pd.read_csv(data_path+front_str+'yvalid.csv',header=None)

# transform x from Dataframe (pandas read csv format) format to matrix format
X_train_matrix = X_train_o.as_matrix()
X_valid_matrix = X_valid_o.as_matrix()

# transform y from DataFrame (pandas read csv format) to categorical, from (1 or 2) to (0 1 0) or (0 0 1)
# this is set the value position to 1, anything else to 0
y_train_tocategrical = np_utils.to_categorical(y_train_o)
y_valid_tocategrical = np_utils.to_categorical(y_valid_o)

# select the last two columns of y as there are only two categries from (0 1 0) or (0 0 1) to (1 0) or (0 1)
y_train_select = y_train_tocategrical[:,1:]
y_valid_select = y_valid_tocategrical[:,1:]

# remember the start position of seizures
idx_valid = idx_seizure(y_valid_select)

# normalize the data  ; make sure the feature variance is 1 and the mean is 0
# transform x from 1.0 -> 2.9 to -2 -> 2, most of it will be around -1 to 1
# must use fit_transform for the train set as this is the prior condition of it
# although there is no difference result between fit_transform and transform
scaler = StandardScaler()

# if scaler flag is set then scaler function will not be used
if(FLAGS.scaler):
    X_train_fittransform = scaler.fit_transform(X_train_matrix)
    X_valid_transform = scaler.transform(X_valid_matrix)
else:
    X_train_fittransform = X_train_matrix
    X_valid_transform = X_valid_matrix

# create sequence data for x and y
# x will be transformed from (9336, 24) to (9286, 50, 24)
# y will be transformed from (9336, 2) to (9286, 2)
X_valid_createseq,y_valid_createseq = create_seq(X_valid_transform,y_valid_select,seq_length,overlap,predstep)
X_train_createseq,y_train_createseq = create_seq(X_train_fittransform,y_train_select,seq_length,overlap,predstep)


def sumSingleLineInMatrix(matrix, linenum):
    return np.sum(matrix[:, linenum] == 1)

if y_train_createseq.shape[1] == 2:
    n_preict = sumSingleLineInMatrix(y_train_createseq, 1)
    n_int = sumSingleLineInMatrix(y_train_createseq, 0)
    n_preict_valid = sumSingleLineInMatrix(y_valid_createseq, 1)
    n_int_valid = sumSingleLineInMatrix(y_valid_createseq, 0)
else:
    n_ict = sumSingleLineInMatrix(y_train_createseq, 2)
    n_posict = sumSingleLineInMatrix(y_train_createseq, 3)
    n_ict_valid = sumSingleLineInMatrix(y_valid_createseq, 2)
    n_posict_valid = sumSingleLineInMatrix(y_valid_createseq, 3)

def train_models(params):
    params = params.flatten()
    learning_rate = float(params[0])
    FLAGS.lr = learning_rate
    batch_size = FLAGS.batch_size
    dropouts = FLAGS.dropout
    np_epoch = FLAGS.epochs
    l2weight = FLAGS.l2weight
    optimizer_str = FLAGS.optimizer
    structure = [['lstm',FLAGS.hidden_unit], ['lstm', FLAGS.hidden_unit] ]
    model = build_net(structure, X_train_createseq, y_train_createseq, X_valid_createseq, y_valid_createseq,\
            batch_size, learning_rate, dropouts,np_epoch, l2weight, optimizer_str)
    y_pred = model.predict(X_valid_createseq)
    y_pred, y_valid_fire = firing_power(y_pred, y_valid_createseq, FLAGS.firing_threshold,\
                                   FLAGS.firing_prob, FLAGS.firing_filter_window)
    fpr, sens = performance(y_pred, y_valid_fire, idx_valid)

    endcurrentDT = datetime.datetime.now()
    print("fpr = "+ str(fpr))
    print("sensitivity = "+ str(sens))
    print("Training time	  ----------- " + str(endcurrentDT-currentDT))
    print("save str names: batchfilename, fpr, sens, lr, dropout, validationloss")


    validation_loss = model.evaluate(X_valid_createseq, y_valid_createseq, verbose=0)
    print("valid_loss in func = "+str(validation_loss[0]))
    validation_loss_list.append(validation_loss[0])

    save_str = FLAGS.batch_filename+","+ str("%.4f"%fpr)+ ","+ str("%.4f"% sens)+","+str("%.4f"%FLAGS.lr)+","+str("%.4f"%FLAGS.dropout)+ \
                ","+str("%.4f"%validation_loss[0])
    save_str_app(save_str, FLAGS.result_filename)
    return float(validation_loss[0])

def do_gpy_opt():
    multistep_domain = get_param_domain(FLAGS.tune_parameter)
    lstm_bopt = GPyOpt.methods.BayesianOptimization(train_models,
                                domain=multistep_domain,
                                acquisition_type='EI',
                                exact_feval=True)
    lstm_bopt.run_optimization(FLAGS.bo_epoch)
    print("------------------------params--------------------------------------")
    for i in range(0,len(validation_loss_list)):
        params_list.append(float(lstm_bopt.X[i]))
    print("==Optimum Hyperparams==")
    print (lstm_bopt.x_opt)
    print("Min test loss %f" % min(validation_loss_list))
    endcurrentDT = datetime.datetime.now()
    print("Training End   time----------- " + str(endcurrentDT))
    print("Training time	  ----------- " + str(endcurrentDT-currentDT))
    print("------------------------end bayesian optimisation--------------------------------------")
    ac_png_path = FLAGS.images + FLAGS.batch_filename +"acquisition"
    pngname = FLAGS.images + FLAGS.batch_filename +"convergence"
    lstm_bopt.plot_convergence(pngname)
    lstm_bopt.plot_acquisition(filename=ac_png_path)

do_gpy_opt()

def saveResult(filename, results):
    with open(filename, 'w') as f:
        f.write(str(results))

paramslistfile = "results/params" + FLAGS.batch_filename +".txt"
validationlistfile="results/validloss"+ FLAGS.batch_filename+".txt"

saveResult(paramslistfile,params_list)
saveResult(validationlistfile, validation_loss_list)
