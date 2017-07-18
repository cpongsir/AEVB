import scipy.io as sio
import scipy as sp
import numpy as np
from copy import deepcopy
from sklearn.linear_model import Ridge

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

sys.path.append('lib')

from GenerativeModel import *
from RecognitionModelMod import *
from SGVB import *

"""
Generate dictionary for Generative Model initialization
"""

def gen_model(win_size, xDim, yDim, vDim):

    gen_nn = lasagne.layers.InputLayer((None, xDim))
    gen_nn = lasagne.layers.DenseLayer(gen_nn, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    gen_nn = lasagne.layers.DenseLayer(gen_nn, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())

    gen_nn = lasagne.layers.DenseLayer(gen_nn, yDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_XtoY_Params = dict([('network', gen_nn)])

    gen_nn_v = lasagne.layers.InputLayer((None, win_size*xDim))
    gen_nn_v = lasagne.layers.DenseLayer(gen_nn_v, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    gen_nn_v = lasagne.layers.DenseLayer(gen_nn_v, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())

    gen_nn_v = lasagne.layers.DenseLayer(gen_nn_v, vDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_XtoV_Params = dict([('network', gen_nn_v)])

    gen_nn_v_0to3 = lasagne.layers.InputLayer((None, xDim))
    gen_nn_v_0to3 = lasagne.layers.DenseLayer(gen_nn_v_0to3, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    gen_nn_v_0to3 = lasagne.layers.DenseLayer(gen_nn_v_0to3, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())

    gen_nn_v_0to3 = lasagne.layers.DenseLayer(gen_nn_v_0to3, vDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_XtoV_Params_0to3 = dict([('network', gen_nn_v_0to3)])


    gen_nn_dyn = lasagne.layers.InputLayer((None, xDim))
    gen_nn_dyn = lasagne.layers.DenseLayer(gen_nn_dyn, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    gen_nn_dyn = lasagne.layers.DenseLayer(gen_nn_dyn, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())

    gen_nn_dyn = lasagne.layers.DenseLayer(gen_nn, xDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_dyn_Params = dict([('network', gen_nn_dyn)])

    gendict = dict([('A'     , 0.8*np.eye(xDim)),         # Linear dynamics parameters
                    ('QChol' , 2*np.diag(np.ones(xDim))), # innovation noise
                    ('Q0Chol', 2*np.diag(np.ones(xDim))),
                    ('x0'    , np.zeros(xDim)),
                    ('NN_XtoY_Params',NN_XtoY_Params),    # neural network output mapping
                    ('NN_XtoV_Params',NN_XtoV_Params),
                    ('NN_XtoV_Params_0to3',NN_XtoV_Params_0to3),
                    ('NN_dyn_Params', NN_dyn_Params),
                    ('output_nlin' , 'exponential'),  # for poisson observations
                    ('win_size', win_size)
                    ])

    return gendict

"""
Generate dictionary for Recognition Model initialization
"""

def rec_model(rec, xDim, yDim, vDim):
    if rec == 'SmoothingTimeSeriesRNN' : input_dim = xDim+yDim
    else : input_dim = yDim
    # Describe network for mapping into means for t=0
    NN_Mu0 = lasagne.layers.InputLayer((None, yDim))
    NN_Mu0 = lasagne.layers.DenseLayer(NN_Mu0, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_Mu0 = lasagne.layers.DenseLayer(NN_Mu0, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_Mu0 = lasagne.layers.DenseLayer(NN_Mu0, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_Mu0 = lasagne.layers.DenseLayer(NN_Mu0, xDim, nonlinearity=linear, W=lasagne.init.Normal())
    NN_Mu0.W.set_value(NN_Mu0.W.get_value()*10)
    NN_Mu0 = dict([('network', NN_Mu0), ('is_train', False)])

    # Describe network for mapping into means
    NN_Mu = lasagne.layers.InputLayer((None, input_dim))
    NN_Mu = lasagne.layers.DenseLayer(NN_Mu, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_Mu = lasagne.layers.DenseLayer(NN_Mu, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_Mu = lasagne.layers.DenseLayer(NN_Mu, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_Mu = lasagne.layers.DenseLayer(NN_Mu, xDim, nonlinearity=linear, W=lasagne.init.Normal())
    NN_Mu.W.set_value(NN_Mu.W.get_value()*10)
    NN_Mu = dict([('network', NN_Mu), ('is_train', False)])

    ########################################
    # Describe network for mapping into Covariances
    NN_Lambda = lasagne.layers.InputLayer((None, yDim))
    NN_Lambda = lasagne.layers.DenseLayer(NN_Lambda, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_Lambda = lasagne.layers.DenseLayer(NN_Lambda, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_Lambda = lasagne.layers.DenseLayer(NN_Lambda, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_Lambda = lasagne.layers.DenseLayer(NN_Lambda, xDim*xDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_Lambda.W.set_value(NN_Lambda.W.get_value()*10)
    NN_Lambda = dict([('network', NN_Lambda), ('is_train', False)])


    NN_LambdaX = lasagne.layers.InputLayer((None, 2*yDim))
    NN_LambdaX = lasagne.layers.DenseLayer(NN_LambdaX, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_LambdaX = lasagne.layers.DenseLayer(NN_LambdaX, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_LambdaX = lasagne.layers.DenseLayer(NN_LambdaX, 60, nonlinearity=tanh, W=lasagne.init.Orthogonal())
    NN_LambdaX = lasagne.layers.DenseLayer(NN_LambdaX, xDim*xDim, nonlinearity=linear, W=lasagne.init.Orthogonal())
    NN_LambdaX.W.set_value(NN_LambdaX.W.get_value()*10)
    NN_LambdaX = dict([('network', NN_LambdaX)])

    ########################################
    # define dictionary of recognition model parameters
    recdict = dict([('A'     , .9*np.eye(xDim)),
                    ('QinvChol',  np.eye(xDim)), #np.linalg.cholesky(np.linalg.inv(np.array(tQ)))),
                    ('Q0invChol', np.eye(xDim)), #np.linalg.cholesky(np.linalg.inv(np.array(tQ0)))),
                    ('NN_Mu0' ,NN_Mu0),
                    ('NN_Mu' ,NN_Mu),
                    ('NN_Lambda',NN_Lambda),
                    ('NN_LambdaX',NN_LambdaX),
                    ])

    return recdict

"""
Initialize SGVB_Kinematics instance
"""

def sgvb_init(gen, rec, win_size, xDim, yDim, vDim):
    gendict = gen_model(win_size, xDim, yDim, vDim)
    recdict = rec_model(rec, xDim, yDim, vDim)
    if gen != 'PLDS' :
        sgvb = SGVB_Kinematics(gendict, eval(gen), recdict, eval(rec), xDim = xDim, yDim = yDim, vDim = vDim)
    else :
        sgvb = SGVB(gendict, eval(gen), recdict, eval(rec), xDim = xDim, yDim = yDim)
    sgvb_best = deepcopy(sgvb)
    return sgvb, sgvb_best

"""
Split neural data into train and test for n-fold cross-validation
"""

def split_data(T, numFolds, fold):
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

"""
Calculate r-squared for x-velocuty, y-velocity, and concatenated velocity
"""

def getRSquared(v_test_x, v_test_y, v_test_x_frommodel, v_test_y_frommodel):
    y_i = v_test_x
    y_bar = y_i.mean()
    y_ihat = v_test_x_frommodel
    SST = np.sum((y_i - y_bar)**2)
    SSReg = np.sum((y_ihat - y_i)**2)
    RsquaredX = 1 - SSReg/SST

    y_i = v_test_y
    y_bar = y_i.mean()
    y_ihat = v_test_y_frommodel
    SST = np.sum((y_i - y_bar)**2)
    SSReg = np.sum((y_ihat - y_i)**2)
    RsquaredY = 1 - SSReg/SST

    y_i = np.concatenate((v_test_x, v_test_y))
    y_bar = y_i.mean()
    y_ihat = np.concatenate((v_test_x_frommodel, v_test_y_frommodel))
    SST = np.sum((y_i - y_bar)**2)
    SSReg = np.sum((y_ihat - y_i)**2)
    RsquaredCombined = 1 - SSReg/SST

    return RsquaredX, RsquaredY, RsquaredCombined

def getKinematicsTrainTest(T_train, T_test):
    trainLen = len(T_train)
    testLen = len(T_test)
    v_train_x = np.transpose(T_train[0]['X'][3,:])
    for i in range(1, trainLen):
        v_train_x = np.concatenate((v_train_x, np.transpose(T_train[i]['X'][3,:])), axis=0)

    v_train_y = np.transpose(T_train[0]['X'][4,:])
    for i in range(1, trainLen):
        v_train_y = np.concatenate((v_train_y, np.transpose(T_train[i]['X'][4,:])), axis=0)

    v_test_x = np.transpose(T_test[0]['X'][3,:])
    for i in range(testLen):
        v_test_x = np.concatenate((v_test_x, np.transpose(T_test[i]['X'][3,:])), axis=0)

    v_test_y = np.transpose(T_test[0]['X'][4,:])
    for i in range(testLen):
        v_test_y = np.concatenate((v_test_y, np.transpose(T_test[i]['X'][4,:])), axis=0)

    return v_train_x, v_train_y, v_test_x, v_test_y

"""
Main function that train SGVB model and save results under 'name.p'
"""

def run(sgvb, learning_rate, use_kinematics, T_train, T_test, n_epochs, name):
    if use_kinematics :
        runKinematics(sgvb, learning_rate, T_train, T_test, n_epochs, name)
    else:
        runNonKinematics(sgvb, learning_rate, T_train, T_test, n_epochs, name)

"""
A function that train PLDSK and PNNK
"""

def runKinematics(sgvb, learning_rate, T_train, T_test, n_epochs, name):

    batch_y = Te.matrix('batch_y')
    batch_v = Te.matrix('batch_v')

    trainLen = len(T_train)
    testLen = len(T_test)

    v_train_x, v_train_y, v_test_x, v_test_y = getKinematicsTrainTest(T_train, T_test)

    # sgvb, sgvb_best = sgvb_init(gen, rec, win_size, xDim, yDim, vDim)
    updates = lasagne.updates.adam(-sgvb.cost(), sgvb.getParams(), learning_rate=learning_rate)

    train_fn = theano.function(
             outputs=sgvb.cost(),
             inputs=[theano.In(batch_y), theano.In(batch_v)],
             updates=updates,
             givens={sgvb.Y: batch_y, sgvb.V: batch_v},
        )

    cost = []
    r_train_x = []
    r_train_y = []
    r_train_c = []
    r_test_x = []
    r_test_y = []
    r_test_c = []

    higest_cost = float("-inf")

    count = 0
    c_old = 0

    for ie in np.arange(n_epochs):

        print('--> entering epoch %d' % ie)

        for i in np.arange(0, trainLen, 10):

            y = np.transpose(T_train[i]['Z'])
            v = np.transpose(T_train[i]['X'][3:5,:])

            for j in range(i+1, i+10) :
                if j >= trainLen : break
                y = np.concatenate((y, np.transpose(T_train[j]['Z'])), axis=0)
                v = np.concatenate((v, np.transpose(T_train[j]['X'][3:5,:])), axis=0)
            c = train_fn(y,v*10)
            cost.append(c)
            if c > higest_cost:
                higest_cost = c
                sgvb_best = deepcopy(sgvb)

        postX = sgvb.mrec.postX

        x_train = postX.eval({sgvb.Y: np.transpose(T_train[0]['Z'])})
        for ii in range(1, trainLen):
            x_train = np.concatenate((x_train, postX.eval({sgvb.Y: np.transpose(T_train[ii]['Z'])})), axis=0)
        v_train_x_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_train})[:,0]/10
        v_train_y_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_train})[:,1]/10

        RsquaredX, RsquaredY, RsquaredCombined = getRSquared(v_train_x, v_train_y, v_train_x_frommodel, v_train_y_frommodel)

        r_train_x.append(RsquaredX)
        r_train_y.append(RsquaredY)
        r_train_c.append(RsquaredCombined)

        x_test = postX.eval({sgvb.Y: np.transpose(T_test[0]['Z'])})
        for ii in range(testLen):
            x_test = np.concatenate((x_test, postX.eval({sgvb.Y: np.transpose(T_test[ii]['Z'])})), axis=0)

        v_test_x_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_test})[:,0]/10
        v_test_y_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_test})[:,1]/10

        RsquaredX, RsquaredY, RsquaredCombined = getRSquared(v_test_x, v_test_y, v_test_x_frommodel, v_test_y_frommodel)

        r_test_x.append(RsquaredX)
        r_test_y.append(RsquaredY)
        r_test_c.append(RsquaredCombined)

        # with open('PLDS_main_fold'+str(fold)+'.txt', 'a') as flog:
        #     flog.write('epoch '+str(ie)+' : '+str(c)+' '+str(RsquaredX)+' '+str(RsquaredY)+' '+str(RsquaredCombined)+'\n')
        # flog.close()

        if abs(c-c_old) < 0.1:
            count = count+1
        else:
            count = 0
        c_old = c

        if count > 2: break


    st = {'cost':cost, 'r_train_x':r_train_x, 'r_train_y':r_train_y, 'r_train_c':r_train_c, 'r_test_x':r_test_x, 'r_test_y':r_test_y, 'r_test_c':r_test_c, 'sgvb': sgvb}

    with open(name+'.p', 'wb') as fp:
        pickle.dump(st, fp)
    fp.close()

"""
Perform linear regression
"""
def predictFromRegression(input_train, output_train, input_test):
    clf = Ridge(alpha=1)
    clf.fit(input_train, output_train)
    output_test = clf.predict(input_test)
    return output_test

"""
A function that trains PLDS
"""

def runNonKinematics(sgvb, learning_rate, T_train, T_test, n_epochs, name):

    batch_y = Te.matrix('batch_y')
    batch_v = Te.matrix('batch_v')

    trainLen = len(T_train)
    testLen = len(T_test)

    v_train_x, v_train_y, v_test_x, v_test_y = getKinematicsTrainTest(T_train, T_test)

    # sgvb, sgvb_best = sgvb_init(gen, rec, win_size, xDim, yDim, vDim)
    updates = lasagne.updates.adam(-sgvb.cost(), sgvb.getParams(), learning_rate=learning_rate)

    train_fn = theano.function(
             outputs=sgvb.cost(),
             inputs=[theano.In(batch_y)],
             updates=updates,
             givens={sgvb.Y: batch_y},
        )

    cost = []
    r_train_x = []
    r_train_y = []
    r_train_c = []
    r_test_x = []
    r_test_y = []
    r_test_c = []

    higest_cost = float("-inf")

    count = 0
    c_old = 0

    for ie in np.arange(n_epochs):

        print('--> entering epoch %d' % ie)

        for i in np.arange(0, trainLen, 10):

            y = np.transpose(T_train[i]['Z'])

            for j in range(i+1, i+10) :
                if j >= trainLen : break
                y = np.concatenate((y, np.transpose(T_train[j]['Z'])), axis=0)

            c = train_fn(y)
            cost.append(c)
            if c > higest_cost:
                higest_cost = c
                sgvb_best = deepcopy(sgvb)

        postX = sgvb.mrec.postX

        x_train = postX.eval({sgvb.Y: np.transpose(T_train[0]['Z'])})
        for ii in range(1, trainLen):
            x_train = np.concatenate((x_train, postX.eval({sgvb.Y: np.transpose(T_train[ii]['Z'])})), axis=0)

        v_train_x_frommodel = predictFromRegression(x_train, v_train_x, x_train)
        v_train_y_frommodel = predictFromRegression(x_train, v_train_y, x_train)

        # v_train_x_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_train})[:,0]/10
        # v_train_y_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_train})[:,1]/10

        RsquaredX, RsquaredY, RsquaredCombined = getRSquared(v_train_x, v_train_y, v_train_x_frommodel, v_train_y_frommodel)

        r_train_x.append(RsquaredX)
        r_train_y.append(RsquaredY)
        r_train_c.append(RsquaredCombined)

        x_test = postX.eval({sgvb.Y: np.transpose(T_test[0]['Z'])})
        for ii in range(testLen):
            x_test = np.concatenate((x_test, postX.eval({sgvb.Y: np.transpose(T_test[ii]['Z'])})), axis=0)

        v_test_x_frommodel = predictFromRegression(x_train, v_train_x, x_test)
        v_test_y_frommodel = predictFromRegression(x_train, v_train_y, x_test)


        # v_test_x_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_test})[:,0]/10
        # v_test_y_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_test})[:,1]/10

        RsquaredX, RsquaredY, RsquaredCombined = getRSquared(v_test_x, v_test_y, v_test_x_frommodel, v_test_y_frommodel)

        r_test_x.append(RsquaredX)
        r_test_y.append(RsquaredY)
        r_test_c.append(RsquaredCombined)

        # with open('PLDS_main_fold'+str(fold)+'.txt', 'a') as flog:
        #     flog.write('epoch '+str(ie)+' : '+str(c)+' '+str(RsquaredX)+' '+str(RsquaredY)+' '+str(RsquaredCombined)+'\n')
        # flog.close()

        if abs(c-c_old) < 0.1:
            count = count+1
        else:
            count = 0
        c_old = c

        if count > 2: break


    st = {'cost':cost, 'r_train_x':r_train_x, 'r_train_y':r_train_y, 'r_train_c':r_train_c, 'r_test_x':r_test_x, 'r_test_y':r_test_y, 'r_test_c':r_test_c, 'sgvb': sgvb}

    with open(name+'.p', 'wb') as fp:
        pickle.dump(st, fp)
    fp.close()

    # v_test_x_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_test})[:,0]/10
    # v_test_y_frommodel = sgvb.mprior.vrate.eval({sgvb.mprior.Xsamp: x_test})[:,1]/10
    #
    # RsquaredX, RsquaredY, RsquaredCombined = getRSquared(rec, v_test_x, v_test_y, v_test_x_frommodel, v_test_y_frommodel)
    #
    #
    # with open('PLDS_main.txt', 'a') as f:
    #     f.write('fold '+str(fold)+' : '+str(RsquaredX) + ' '+str(RsquaredY) + ' '+str(RsquaredCombined) + '\n')
    # f.close()
