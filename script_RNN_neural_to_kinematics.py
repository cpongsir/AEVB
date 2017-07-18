import scipy.io as sio
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import theano
import theano.tensor as Te
import theano.tensor.nlinalg as Tla
import lasagne      
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify, sigmoid
from theano.tensor.shared_randomstreams import RandomStreams
from numpy.random import *
from matplotlib import pyplot as plt
import cPickle as pickle
import sys

from keras.layers import LSTM, Dense, Masking
from keras.models import Sequential
from keras import optimizers

def getInputOutput(T, time_max):
    timesteps = []

    inputs_array = []
    for i in range(len(T)):
        timesteps.append(T[i]['Z'].T.shape[0])
        inputs = np.concatenate((T[i]['Z'].T, np.zeros((time_max - T[i]['Z'].T.shape[0], T[i]['Z'].T.shape[1]))), axis=0)
        inputs_array.append(inputs)
    inputs = np.stack(inputs_array, axis=0)
    
    
    outputs_array = []
    for i in range(len(T)):
        outputs = np.concatenate((T[i]['X'][3:5,:].T, np.zeros((time_max - T[i]['X'][3:5,:].T.shape[0], T[i]['X'][3:5,:].T.shape[1]))), axis=0)
        outputs_array.append(outputs)
    outputs = np.stack(outputs_array, axis=0)
    
    return inputs, outputs, timesteps

def split_data(T, numFolds, fold):
    ## split T into T_train and T_test
    ## train = 0:399
    ## test = 400:end
    
    lengthEach = int(T.size / numFolds)
    testRange = range(fold*lengthEach, (fold+1)*lengthEach)   
    
    T_train = []
    T_test = []
    
    for i in range(T.size):
        if i in testRange:
            T_test.append(T[0,i])
        else:
            T_train.append(T[0,i])    
    return T_train, T_test

def getRSquaredStack(y, y_hat, timesteps):
    
    numTrials = y.shape[0]
    
    y_all = y[0,0:timesteps[0],:]
    y_hat_all = y_hat[0,0:timesteps[0],:]
    
    for i in range(1, numTrials):
        y_all = np.concatenate((y_all, y[i,0:timesteps[i],:]), axis=0)
        y_hat_all = np.concatenate((y_hat_all, y_hat[i,0:timesteps[i],:]), axis=0)
        
    y_all = y_all.reshape((1,y_all.size))
    y_hat_all = y_hat_all.reshape((1,y_hat_all.size))
    
    y_i = y_all
    y_ihat = y_hat_all
    y_bar = y_i.mean()
    SST = np.sum((y_i - y_bar)**2)
    SSReg = np.sum((y_ihat - y_i)**2)

    return 1 - SSReg/SST

# load the T-struct
# TT.mat contains the first 1000 trials from R_2017-03-21.mat
T = sio.loadmat('TT.mat')['TT']

# time_max is the maximum number of timesteps in each trial.
# Inputs will be zero-padded to have the length equal to time_max. 
# Inputs that are longer than time_max will be clipped to have the length equal to time_max.
time_max = 150

n_epochs = 100
batch_size = 32
learning_rate = 0.001
decay_rate = 1e-6
dropout_rate = 0.5

xDim = 20
yDim = 192
vDim = 2

num_nodes = 32

for fold in range(10):
    
    print 'fold', fold
    
    name = 'LSTM_fold'+str(fold)
    
    T_train, T_test = split_data(T, 10, fold)
    inputsTrain, outputsTrain, timestepsTrain = getInputOutput(T_train, time_max)
    inputsTest, outputsTest, timestepsTest = getInputOutput(T_test, time_max)
    
    batch_size=128
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(time_max, yDim)))
    model.add(LSTM(num_nodes, input_dim=yDim, dropout=dropout_rate, return_sequences=True))
    model.add(LSTM(num_nodes, dropout=dropout_rate, return_sequences=True))
    model.add(LSTM(vDim, dropout=dropout_rate, return_sequences=True))
    opt = optimizers.adam(lr=learning_rate, decay=decay_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    
    model.fit(inputsTrain, outputsTrain, epochs=n_epochs, batch_size=batch_size, verbose=0)
    
    outputsTrainPredict = model.predict(inputsTrain)
    outputsTestPredict = model.predict(inputsTest)
    
    rSqauredTrain = getRSquaredStack(outputsTrain, outputsTrainPredict, timestepsTrain)
    rSquaredTest = getRSquaredStack(outputsTest, outputsTestPredict, timestepsTest)
    
    st = {'r_train': rSqauredTrain, 'r_test': rSquaredTest, 'model': model}
    
    with open(name+'.p', 'wb') as fp:
        pickle.dump(st, fp)
    fp.close()
    
    """
    with open('RNN_neural_to_kinematics.txt', 'a') as f:
        f.write('fold '+str(fold)+' '+str(rSqauredTrain)+' '+str(rSquaredTest)+'\n')
    f.close()
    """
    