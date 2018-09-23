# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
import math

def idx_seizure(y, sop, len_sop):
  """
  Inputs
    y - target matrix
  Outputs
    idx - indexes of where each seizure begins and ends
  """
  idx = []
  for i in range(0,y.shape[0]):
    if y[i,1] == 1 and y[i-1,1] == 0 or (i==0 and y[0,1] == 1):
      large_length = change_sop_to_timestep(sop)*len_sop
      for j in range(i, int(i+large_length+1)):
        try:
          if y[j,1] == 1 and y[j+1,1] == 0:
            idx.append([i,j])
            break
        except IndexError:
          idx.append([i,j])
          break
  return idx

def get_y(data):
    return data[:,-1]

def get_X(data):
    return data[:, :-1]

# set pre_ictal (SOP period) time to 1, else to 0
def arrange_y_withsop(y, sop):
    y_categrical = get_y_tocategrical(y)
    idx = idx_seizure(y_categrical, sop, 1)
    print("idx before: \n %s"% str(idx))
    y_ = np.zeros(len(y))
    sop_period_num = change_sop_to_timestep(sop)
    for i in range(len(idx)):
        end = idx[i][0]
        start = end - sop_period_num
        if(start < 0):
            start = 0
        for j in range(sop_period_num):
            y_[start+j] = 1
    return y_
    
# As our feature window is 5 second, this function
# is to convert minutes to time step numbers
def change_sop_to_timestep(sop):
    return int(sop*60/5)

def get_y_tocategrical(y):
    y_tocategrical = np_utils.to_categorical(y)
    return y_tocategrical

def separate_data(X,y,ptr,pv,idx):
  """
  Inputs
    X - feature matrix
    y - target matrix
    ptr - percentage of seizures included in the training set
    pv - percentage of seizures included in the validation set
    idx - indexes of where each seizure begins and ends
  
  Outputs
    X_train - train feature matrix 
    y_train - train target matrix
    X_valid - validation feature matrix 
    y_valid - validation target matrix
    X_test - test feature matrix 
    y_test - test target matrix
  """

  ntr = int(len(idx)*ptr)
  nt  = ntr + int(len(idx)*pv)
  print("train_nums = %d total_train_valid_nums=%d"%(ntr, nt))
  vtr = int(idx[ntr-1][1]+1)
  vt  = int(idx[nt-1][1]+1)
  print("vtr= %d vt=%d"%(vtr, vt))
  X_train = X[:vtr,:]
  X_valid = X[vtr:vt,:]
  X_test =  X[vt:,:]
  check_nan(X_train)
  check_nan(X_valid)
  check_nan(X_test)
  X_train_fittransform = get_scaler_fit(X_train)
  X_valid_transform = get_scaler(X_valid)
  X_test_transform = get_scaler(X_test)
  X_test =  X[vt:,:]
  y_train = y[:vtr,:]
  y_valid = y[vtr:vt,:]
  y_test =  y[vt:,:]
#  return X_train, X_valid, X_test, y_train, y_valid, y_test

  return X_train_fittransform, X_valid_transform, X_test_transform, y_train, y_valid, y_test

scaler = StandardScaler()


def get_scaler_fit(X):
  return scaler.fit_transform(X)

def get_scaler(X):
  return   scaler.transform(X)

def check_nan(X):
  """
  Inputs
    X - feature matrix
    y - target matrix
  Outputs
    X - feature matrix
    y - target matrix
  """
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      if math.isnan(X[i,j]):
        X[i,j]=0
  return X 

def get_train_valid_test(data_path, patient_id_str, sop, \
                             train_percent, valid_percent):
        
    patient_id = patient_id_str
    data_path = data_path
    sop = sop
    whole_data = pd.read_csv(data_path+patient_id+'_eeg.csv',header=None)
    whole_data_matrix = whole_data.as_matrix()
    ori_y = get_y(whole_data_matrix)
    y_arrage = arrange_y_withsop(ori_y, sop)
    y_arrage_tocategrical = get_y_tocategrical(y_arrage)
    X_whole = get_X(whole_data_matrix)
    idx = idx_seizure(y_arrage_tocategrical, sop, 1)
    print("idx now: \n %s"% str(idx))
    return separate_data(X_whole, y_arrage_tocategrical,train_percent, valid_percent, idx)
    

