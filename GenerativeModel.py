"""
The MIT License (MIT)
Copyright (c) 2015 Evan Archer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import theano
import lasagne
import theano.tensor as T
import theano.tensor.nlinalg as Tla
import theano.tensor.slinalg as Tsla
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import pdb

class GenerativeModel(object):
    '''
    Interface class for generative time-series models
    '''
    def __init__(self,GenerativeParams,xDim,yDim, A, QinvChol, Q0invChol, srng = None,nrng = None):

        # input variable referencing top-down or external input

        self.xDim = xDim
        self.yDim = yDim

        self.srng = srng
        self.nrng = nrng

        # internal RV for generating sample
        self.Xsamp = T.matrix('Xsamp')

    def evaluateLogDensity(self):
        '''
        Return a theano function that evaluates the density of the GenerativeModel.
        '''
        raise Exception('Cannot call function of interface class')

    def getParams(self):
       
        '''
        Return parameters of the GenerativeModel.
        '''
        raise Exception('Cannot call function of interface class')

    def generateSamples(self):
        '''
        generates joint samples
        '''
        raise Exception('Cannot call function of interface class')

    def __repr__(self):
        return "GenerativeModel"

class LDS(GenerativeModel):
    '''
    Gaussian latent LDS with (optional) NN observations:

    x(0) ~ N(x0, Q0 * Q0')
    x(t) ~ N(A x(t-1), Q * Q')
    y(t) ~ N(NN(x(t)), R * R')

    For a Kalman Filter model, choose the observation network, NN(x), to be
    a one-layer network with a linear output. The latent state has dimensionality
    n (parameter "xDim") and observations have dimensionality m (parameter "yDim").

    Inputs:
    (See GenerativeModel abstract class definition for a list of standard parameters.)

    GenerativeParams  -  Dictionary of LDS parameters
                           * A     : [n x n] linear dynamics matrix; should
                                     have eigenvalues with magnitude strictly
                                     less than 1
                           * QChol : [n x n] square root of the innovation
                                     covariance Q
                           * Q0Chol: [n x n] square root of the innitial innovation
                                     covariance
                           * RChol : [n x 1] square root of the diagonal of the
                                     observation covariance
                           * x0    : [n x 1] mean of initial latent state
                           * NN_XtoY_Params:
                                   Dictionary with one field:
                                    - network: a lasagne network with input
                                      dimensionality n and output dimensionality m
    '''
    def __init__(self, GenerativeParams, xDim, yDim, A, QChol, Q0Chol, srng = None, nrng = None):

        super(LDS, self).__init__(GenerativeParams,xDim,yDim,A, QChol, Q0Chol,srng,nrng)

        # parameters
        """
        if 'A' in GenerativeParams:
            self.A      = theano.shared(value=GenerativeParams['A'].astype(theano.config.floatX), name='A'     ,borrow=True)     # dynamics matrix
        else:
            # TBD:MAKE A BETTER WAY OF SAMPLING DEFAULT A
            self.A      = theano.shared(value=.5*np.diag(np.ones(xDim).astype(theano.config.floatX)), name='A'     ,borrow=True)     # dynamics matrix

        if 'QChol' in GenerativeParams:
            self.QChol  = theano.shared(value=GenerativeParams['QChol'].astype(theano.config.floatX), name='QChol' ,borrow=True)     # cholesky of innovation cov matrix
        else:
            self.QChol  = theano.shared(value=(np.eye(xDim)).astype(theano.config.floatX), name='QChol' ,borrow=True)     # cholesky of innovation cov matrix

        if 'Q0Chol' in GenerativeParams:
            self.Q0Chol = theano.shared(value=GenerativeParams['Q0Chol'].astype(theano.config.floatX), name='Q0Chol',borrow=True)     # cholesky of starting distribution cov matrix
        else:
            self.Q0Chol = theano.shared(value=(np.eye(xDim)).astype(theano.config.floatX), name='Q0Chol',borrow=True)     # cholesky of starting distribution cov matrix
        """
        self.A = A
        self.QChol = QChol
        self.Q0Chol = Q0Chol

        """
        if 'RChol' in GenerativeParams:
            self.RChol  = theano.shared(value=np.ndarray.flatten(GenerativeParams['RChol'].astype(theano.config.floatX)), name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.RChol  = theano.shared(value=np.random.randn(yDim).astype(theano.config.floatX)/10, name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix
        """
        self.RChol  = theano.shared(value=np.random.randn(yDim).astype(theano.config.floatX)/10, name='RChol' ,borrow=True) 

        if 'x0' in GenerativeParams:
            self.x0     = theano.shared(value=GenerativeParams['x0'].astype(theano.config.floatX), name='x0'    ,borrow=True)     # set to zero for stationary distribution
        else:
            self.x0     = theano.shared(value=np.zeros((xDim,)).astype(theano.config.floatX), name='x0'    ,borrow=True)     # set to zero for stationary distribution

        if 'NN_XtoY_Params' in GenerativeParams:
            self.NN_XtoY = GenerativeParams['NN_XtoY_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_XtoY = lasagne.layers.DenseLayer(gen_nn, yDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())

        # set to our lovely initial values
        if 'C' in GenerativeParams:
            self.NN_XtoY.W.set_value(GenerativeParams['C'].astype(theano.config.floatX))
        if 'd' in GenerativeParams:
            self.NN_XtoY.b.set_value(GenerativeParams['d'].astype(theano.config.floatX))

        # we assume diagonal covariance (RChol is a vector)
        self.Rinv    = 1./(self.RChol**2) #Tla.matrix_inverse(T.dot(self.RChol ,T.transpose(self.RChol)))
        self.Lambda  = Tla.matrix_inverse(T.dot(self.QChol ,self.QChol.T))
        self.Lambda0 = Tla.matrix_inverse(T.dot(self.Q0Chol,self.Q0Chol.T))

        # Call the neural network output a rate, basically to keep things consistent with the PLDS class
        self.rate = lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp)

    def sampleX(self, _N):
        _x0 = np.asarray(self.x0.eval(), dtype=theano.config.floatX)
        _Q0Chol = np.asarray(self.Q0Chol.eval(), dtype=theano.config.floatX)
        _QChol = np.asarray(self.QChol.eval(), dtype=theano.config.floatX)
        _A = np.asarray(self.A.eval(), dtype=theano.config.floatX)

        norm_samp = np.random.randn(_N, self.xDim).astype(theano.config.floatX)
        x_vals = np.zeros([_N, self.xDim]).astype(theano.config.floatX)

        x_vals[0] = _x0 + np.dot(norm_samp[0],_Q0Chol.T)

        for ii in xrange(_N-1):
            x_vals[ii+1] = x_vals[ii].dot(_A.T) + norm_samp[ii+1].dot(_QChol.T)

        return x_vals.astype(theano.config.floatX)

    def sampleY(self):
        ''' Return a symbolic sample from the generative model. '''
        return self.rate+T.dot(self.srng.normal([self.Xsamp.shape[0],self.yDim]),T.diag(self.RChol).T)

    def sampleXY(self, _N):
        ''' Return numpy samples from the generative model. '''
        X = self.sampleX(_N)
        nprand = np.random.randn(X.shape[0],self.yDim).astype(theano.config.floatX)
        _RChol = np.asarray(self.RChol.eval(), dtype=theano.config.floatX)
        Y = self.rate.eval({self.Xsamp: X}) + np.dot(nprand,np.diag(_RChol).T)
        return [X,Y]

    def getParams(self):
        return [self.A] + [self.QChol] + [self.Q0Chol] + [self.RChol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY)

    def evaluateLogDensity(self,X,Y):
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        resX  = X[1:]-T.dot(X[:(X.shape[0]-1)],self.A.T)
        resX0 = X[0]-self.x0

        LogDensity  = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum() - (0.5*T.dot(resX.T,resX)*self.Lambda).sum() - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T)
        LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y.shape[0] + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0))  - 0.5*(self.xDim + self.yDim)*np.log(2*np.pi)*Y.shape[0]

        return LogDensity

class PLDS(LDS):
    '''
    Gaussian linear dynamical system with Poisson count observations. Inherits Gaussian
    linear dynamics sampling code from the LDS; implements a Poisson density evaluation
    for discrete (count) data.
    '''
    def __init__(self, GenerativeParams, xDim, yDim, A, QChol, Q0Chol, srng = None, nrng = None):
        # The LDS class expects "RChol" for Gaussian observations - we just pass a dummy
        GenerativeParams['RChol'] = np.ones(1)
        super(PLDS, self).__init__(GenerativeParams,xDim,yDim,A, QChol, Q0Chol,srng,nrng)

        # Currently we emulate a PLDS by having an exponential output nonlinearity.
        # Next step will be to generalize this to more flexible output nonlinearities...
        if GenerativeParams['output_nlin'] == 'exponential':
            self.rate = T.exp(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'sigmoid':
            self.rate = T.nnet.sigmoid(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'softplus':
            self.rate = T.nnet.softplus(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        else:
            raise Exception('Unknown output nonlinearity specification!')

    def getParams(self):
        return [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY)
        #return [self.A] + [self.QChol] + [self.Q0Chol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY)
    
    def getDynParams(self):
        return [self.A] + [self.QChol] + [self.Q0Chol] 

    def sampleY(self):
        ''' Return a symbolic sample from the generative model. '''
        return self.srng.poisson(lam = self.rate, size = self.rate.shape)

    def sampleXY(self,_N):
        ''' Return real-valued (numpy) samples from the generative model. '''
        X = self.sampleX(_N)

        Y = np.random.poisson(lam = self.rate.eval({self.Xsamp: X}))
        return [X.astype(theano.config.floatX),Y.astype(theano.config.floatX)]

    def evaluateLogDensity(self,X,Y):
        # This is the log density of the generative model (*not* negated)
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        resX  = X[1:]-T.dot(X[:-1],self.A.T)
        resX0 = X[0]-self.x0
        LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        #LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        PoisDensity = T.sum(Y * T.log(Ypred)  - Ypred - T.gammaln(Y + 1))
        LogDensity = LatentDensity + PoisDensity
        return LogDensity
    
class LDSKinematics(LDS):
    '''
    Gaussian linear dynamical system with Poisson count observations. Inherits Gaussian
    linear dynamics sampling code from the LDS; implements a Poisson density evaluation
    for discrete (count) data.
    '''
    def __init__(self, GenerativeParams, xDim, yDim, A, QChol, Q0Chol, srng = None, nrng = None):
        # dimension of kinematics
        
        
        # The LDS class expects "RChol" for Gaussian observations - we just pass a dummy
        GenerativeParams['RChol'] = np.ones(1)
        super(LDSKinematics, self).__init__(GenerativeParams,xDim,yDim,A, QChol, Q0Chol,srng,nrng)
        self.vDim = 2
            
        if 'NN_XtoV_Params' in GenerativeParams:
            self.NN_XtoV = GenerativeParams['NN_XtoV_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_XtoV = lasagne.layers.DenseLayer(gen_nn, vDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())
            
        if 'vRChol' in GenerativeParams:
            self.vRChol  = theano.shared(value=np.ndarray.flatten(GenerativeParams['vRChol'].astype(theano.config.floatX)), name='vRChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.vRChol  = theano.shared(value=np.random.randn(self.vDim).astype(theano.config.floatX)/10, name='vRChol' ,borrow=True)
            
        self.vRinv    = 1./(self.vRChol**2)
        self.vrate = lasagne.layers.get_output(self.NN_XtoV, inputs = self.Xsamp)
        
        self.RChol  = theano.shared(value=np.random.randn(self.yDim).astype(theano.config.floatX)/10, name='RChol' ,borrow=True)
        self.Rinv = 1./(self.RChol**2) 

    def getParams(self):
        return [self.A] + [self.QChol] + [self.Q0Chol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY) + [self.vRChol] + lasagne.layers.get_all_params(self.NN_XtoV) 
    
    def sampleV(self):
        return self.vrate+T.dot(self.srng.normal([self.Xsamp.shape[0],self.vDim]),T.diag(self.vRChol).T)
    
    def evaluateLogDensity(self,X,Y,V):
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        resX  = X[1:]-T.dot(X[:(X.shape[0]-1)],self.A.T)
        resX0 = X[0]-self.x0
        Vpred = theano.clone(self.vrate,replace={self.Xsamp: X})
        resV = V - Vpred

        """
        LogDensity  = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum() - (0.5*T.dot(resX.T,resX)*self.Lambda).sum() - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T)
        LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y.shape[0] + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0))  - 0.5*(self.xDim + self.yDim)*np.log(2*np.pi)*Y.shape[0]
        
        KinDensity = -(0.5*T.dot(resV.T,resV)*T.diag(self.vRinv)).sum() + 0.5*(T.log(self.vRinv)).sum()*Y.shape[0] - 0.5*(self.vDim)*np.log(2*np.pi)*Y.shape[0]
        LogDensity += 0*KinDensity
        
        return LogDensity
        """
        LatentDensity = -0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        ObsDensity = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum() 
        # ObsDensity = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum() + 0.5*(T.log(self.Rinv)).sum()*Y.shape[0] - 0.5*(self.yDim)*np.log(2*np.pi)*Y.shape[0]
        KinDensity = -(0.5*T.dot(resV.T,resV)*T.diag(self.vRinv)).sum() + 0.5*(T.log(self.vRinv)).sum()*Y.shape[0] - 0.5*(self.vDim)*np.log(2*np.pi)*Y.shape[0]
        LogDensity = LatentDensity + ObsDensity + KinDensity
        return LogDensity

    
class PLDSKinematics(LDS):
    '''
    Gaussian linear dynamical system with Poisson count observations. Inherits Gaussian
    linear dynamics sampling code from the LDS; implements a Poisson density evaluation
    for discrete (count) data.
    '''
    def __init__(self, GenerativeParams, xDim, yDim, A, QChol, Q0Chol, srng = None, nrng = None):
        # dimension of kinematics
        
        
        # The LDS class expects "RChol" for Gaussian observations - we just pass a dummy
        GenerativeParams['RChol'] = np.ones(1)
        super(PLDSKinematics, self).__init__(GenerativeParams,xDim,yDim,A, QChol, Q0Chol,srng,nrng)
        self.vDim = 2

        # Currently we emulate a PLDS by having an exponential output nonlinearity.
        # Next step will be to generalize this to more flexible output nonlinearities...
        if GenerativeParams['output_nlin'] == 'exponential':
            self.rate = T.exp(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'sigmoid':
            self.rate = T.nnet.sigmoid(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'softplus':
            self.rate = T.nnet.softplus(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        else:
            raise Exception('Unknown output nonlinearity specification!')
            
        if 'NN_XtoV_Params' in GenerativeParams:
            self.NN_XtoV = GenerativeParams['NN_XtoV_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_XtoV = lasagne.layers.DenseLayer(gen_nn, vDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())
            
        if 'vRChol' in GenerativeParams:
            self.vRChol  = theano.shared(value=np.ndarray.flatten(GenerativeParams['vRChol'].astype(theano.config.floatX)), name='vRChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.vRChol  = theano.shared(value=np.random.randn(self.vDim).astype(theano.config.floatX)/10, name='vRChol' ,borrow=True)
            
        self.vRinv    = 1./(self.vRChol**2)
        self.vrate = lasagne.layers.get_output(self.NN_XtoV, inputs = self.Xsamp)

    def getParams(self):
        return [self.A] + [self.QChol] + [self.Q0Chol] + [self.vRChol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY) + lasagne.layers.get_all_params(self.NN_XtoV)

    def sampleY(self):
        ''' Return a symbolic sample from the generative model. '''
        return self.srng.poisson(lam = self.rate, size = self.rate.shape)
    
    def sampleV(self):
        return self.vrate+T.dot(self.srng.normal([self.Xsamp.shape[0],self.vDim]),T.diag(self.vRChol).T)

    def sampleXY(self,_N):
        ''' Return real-valued (numpy) samples from the generative model. '''
        X = self.sampleX(_N)

        Y = np.random.poisson(lam = self.rate.eval({self.Xsamp: X}))
        return [X.astype(theano.config.floatX),Y.astype(theano.config.floatX)]

    def evaluateLogDensity(self,X,Y,V):
        # This is the log density of the generative model (*not* negated)
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        resX  = X[1:]-T.dot(X[:-1],self.A.T)
        resX0 = X[0]-self.x0
        Vpred = theano.clone(self.vrate,replace={self.Xsamp: X})
        resV = V - Vpred
        LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        #LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        PoisDensity = T.sum(Y * T.log(Ypred)  - Ypred - T.gammaln(Y + 1))
        KinDensity = -(0.5*T.dot(resV.T,resV)*T.diag(self.vRinv)).sum() + 0.5*(T.log(self.vRinv)).sum()*Y.shape[0] - 0.5*(self.vDim)*np.log(2*np.pi)*Y.shape[0]
        LogDensity = LatentDensity + PoisDensity + KinDensity
        return LogDensity
    
class PNNKinematics(LDS):
    '''
    Gaussian linear dynamical system with Poisson count observations. Inherits Gaussian
    linear dynamics sampling code from the LDS; implements a Poisson density evaluation
    for discrete (count) data.
    '''
    def __init__(self, GenerativeParams, xDim, yDim, A, QChol, Q0Chol, srng = None, nrng = None):
        # dimension of kinematics
        
        
        # The LDS class expects "RChol" for Gaussian observations - we just pass a dummy
        GenerativeParams['RChol'] = np.ones(1)
        super(PNNKinematics, self).__init__(GenerativeParams,xDim,yDim,A, QChol, Q0Chol,srng,nrng)
        self.vDim = 2

        # Currently we emulate a PLDS by having an exponential output nonlinearity.
        # Next step will be to generalize this to more flexible output nonlinearities...
        if GenerativeParams['output_nlin'] == 'exponential':
            self.rate = T.exp(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'sigmoid':
            self.rate = T.nnet.sigmoid(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'softplus':
            self.rate = T.nnet.softplus(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        else:
            raise Exception('Unknown output nonlinearity specification!')
            
        if 'NN_XtoV_Params' in GenerativeParams:
            self.NN_XtoV = GenerativeParams['NN_XtoV_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_XtoV = lasagne.layers.DenseLayer(gen_nn, vDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())
            
        if 'vRChol' in GenerativeParams:
            self.vRChol  = theano.shared(value=np.ndarray.flatten(GenerativeParams['vRChol'].astype(theano.config.floatX)), name='vRChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.vRChol  = theano.shared(value=np.random.randn(self.vDim).astype(theano.config.floatX)/10, name='vRChol' ,borrow=True)
            
        self.vRinv    = 1./(self.vRChol**2)
        self.vrate = lasagne.layers.get_output(self.NN_XtoV, inputs = self.Xsamp)
        
        if 'NN_dyn_Params' in GenerativeParams:
            self.NN_dyn = GenerativeParams['NN_dyn_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_dyn = lasagne.layers.DenseLayer(gen_nn, xDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())

    def getParams(self):
        return [self.QChol] + [self.Q0Chol] + [self.vRChol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY) + lasagne.layers.get_all_params(self.NN_XtoV) + lasagne.layers.get_all_params(self.NN_dyn)
    
    def sampleX(self, _N):
        _x0 = np.asarray(self.x0.eval(), dtype=theano.config.floatX)
        _Q0Chol = np.asarray(self.Q0Chol.eval(), dtype=theano.config.floatX)
        _QChol = np.asarray(self.QChol.eval(), dtype=theano.config.floatX)
        _A = np.asarray(self.A.eval(), dtype=theano.config.floatX)

        norm_samp = np.random.randn(_N, self.xDim).astype(theano.config.floatX)
        x_vals = np.zeros([_N, self.xDim]).astype(theano.config.floatX)
        
        x_vals[0] = _x0 + np.dot(norm_samp[0],_Q0Chol.T)

        for ii in xrange(_N-1):
            # x_vals[ii+1] = x_vals[ii].dot(_A.T) + norm_samp[ii+1].dot(_QChol.T)
            x_vals[ii+1] = lasagne.layers.get_output(self.NN_dyn, inputs = x_vals[ii]).eval() + norm_samp[ii+1].dot(_QChol.T)

        return x_vals.astype(theano.config.floatX)


    def sampleY(self):
        ''' Return a symbolic sample from the generative model. '''
        return self.srng.poisson(lam = self.rate, size = self.rate.shape)
    
    def sampleV(self):
        return self.vrate+T.dot(self.srng.normal([self.Xsamp.shape[0],self.vDim]),T.diag(self.vRChol).T)

    def sampleXY(self,_N):
        ''' Return real-valued (numpy) samples from the generative model. '''
        X = self.sampleX(_N)

        Y = np.random.poisson(lam = self.rate.eval({self.Xsamp: X}))
        return [X.astype(theano.config.floatX),Y.astype(theano.config.floatX)]

    def evaluateLogDensity(self,X,Y,V):
        # This is the log density of the generative model (*not* negated)
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        # resX  = X[1:]-T.dot(X[:-1],self.A.T)
        resX  = X[1:] - lasagne.layers.get_output(self.NN_dyn, inputs = X[:(X.shape[0]-1)])
        resX0 = X[0]-self.x0
        Vpred = theano.clone(self.vrate,replace={self.Xsamp: X})
        resV = V - Vpred
        LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        #LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        PoisDensity = T.sum(Y * T.log(Ypred)  - Ypred - T.gammaln(Y + 1))
        KinDensity = -(0.5*T.dot(resV.T,resV)*T.diag(self.vRinv)).sum() + 0.5*(T.log(self.vRinv)).sum()*Y.shape[0] - 0.5*(self.vDim)*np.log(2*np.pi)*Y.shape[0]
        LogDensity = LatentDensity + PoisDensity + KinDensity
        return LogDensity
    
class PNNKinematicsWindow(LDS):
    '''
    Gaussian linear dynamical system with Poisson count observations. Inherits Gaussian
    linear dynamics sampling code from the LDS; implements a Poisson density evaluation
    for discrete (count) data.
    '''
    def __init__(self, GenerativeParams, xDim, yDim, A, QChol, Q0Chol, srng = None, nrng = None):
        # dimension of kinematics


        # The LDS class expects "RChol" for Gaussian observations - we just pass a dummy
        GenerativeParams['RChol'] = np.ones(1)
        super(PNNKinematicsWindow, self).__init__(GenerativeParams,xDim,yDim,A, QChol, Q0Chol,srng,nrng)
        self.vDim = 2;

        self.win_size = GenerativeParams['win_size']

        # Currently we emulate a PLDS by having an exponential output nonlinearity.
        # Next step will be to generalize this to more flexible output nonlinearities...
        if GenerativeParams['output_nlin'] == 'exponential':
            self.rate = T.exp(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'sigmoid':
            self.rate = T.nnet.sigmoid(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'softplus':
            self.rate = T.nnet.softplus(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        else:
            raise Exception('Unknown output nonlinearity specification!')
            
        if 'NN_XtoV_Params' in GenerativeParams:
            self.NN_XtoV = GenerativeParams['NN_XtoV_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_XtoV = lasagne.layers.DenseLayer(gen_nn, vDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())

        if 'NN_dyn_Params' in GenerativeParams:
            self.NN_dyn = GenerativeParams['NN_dyn_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_dyn = lasagne.layers.DenseLayer(gen_nn, xDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())

        if 'NN_XtoV_Params_0to3' in GenerativeParams:
            self.NN_XtoV_0to3 = GenerativeParams['NN_XtoV_Params_0to3']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_XtoV_0to3 = lasagne.layers.DenseLayer(gen_nn, vDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())

        if 'vRChol' in GenerativeParams:
            self.vRChol  = theano.shared(value=np.ndarray.flatten(GenerativeParams['vRChol'].astype(theano.config.floatX)), name='vRChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.vRChol  = theano.shared(value=np.random.randn(self.vDim).astype(theano.config.floatX)/10, name='vRChol' ,borrow=True)

        self.vRinv    = 1./(self.vRChol**2)
        
        # self.vrate_0to3 = lasagne.layers.get_output(self.NN_XtoV_0to3, inputs = self.Xsamp[0:4])
        # self.vrate_rest = lasagne.layers.get_output(self.NN_XtoV, inputs = T.concatenate([self.Xsamp[:-4], self.Xsamp[1:-3], self.Xsamp[2:-2], self.Xsamp[3:-1], self.Xsamp[4:]], axis=1))

        ## win_size = 14
        Xsamp14 = T.concatenate([self.Xsamp[:-13], self.Xsamp[1:-12], self.Xsamp[2:-11], self.Xsamp[3:-10], self.Xsamp[4:-9], self.Xsamp[5:-8], self.Xsamp[6:-7], self.Xsamp[7:-6], self.Xsamp[8:-5], self.Xsamp[9:-4], self.Xsamp[10:-3], self.Xsamp[11:-2], self.Xsamp[12:-1], self.Xsamp[13:]], axis=1)

        ## win_size = 12
        Xsamp12 = T.concatenate([self.Xsamp[:-11], self.Xsamp[1:-10], self.Xsamp[2:-9], self.Xsamp[3:-8], self.Xsamp[4:-7], self.Xsamp[5:-6], self.Xsamp[6:-5], self.Xsamp[7:-4], self.Xsamp[8:-3], self.Xsamp[9:-2], self.Xsamp[10:-1], self.Xsamp[11:]], axis=1)

        ## win_size = 10
        Xsamp10 = T.concatenate([self.Xsamp[:-9], self.Xsamp[1:-8], self.Xsamp[2:-7], self.Xsamp[3:-6], self.Xsamp[4:-5], self.Xsamp[5:-4], self.Xsamp[6:-3], self.Xsamp[7:-2], self.Xsamp[8:-1], self.Xsamp[9:]], axis=1)

        ## win_size = 8
        Xsamp8 = T.concatenate([self.Xsamp[:-7], self.Xsamp[1:-6], self.Xsamp[2:-5], self.Xsamp[3:-4], self.Xsamp[4:-3], self.Xsamp[5:-2], self.Xsamp[6:-1], self.Xsamp[7:]], axis=1)

        ## win_size = 6
        Xsamp6 = T.concatenate([self.Xsamp[:-5], self.Xsamp[1:-4], self.Xsamp[2:-3], self.Xsamp[3:-2], self.Xsamp[4:-1], self.Xsamp[5:]], axis=1)

        ## win_size = 4
        Xsamp4 = T.concatenate([self.Xsamp[:-3], self.Xsamp[1:-2], self.Xsamp[2:-1], self.Xsamp[3:]], axis=1)
        
        ## win_size = 2
        Xsamp2 = T.concatenate([self.Xsamp[:-1], self.Xsamp[1:]], axis=1)
        
        if self.win_size == 14 : self.kin_inputs = Xsamp14
        elif self.win_size == 10 : self.kin_inputs = Xsamp10
        elif self.win_size == 8 : self.kin_inputs = Xsamp8
        elif self.win_size == 6 : self.kin_inputs = Xsamp6
        elif self.win_size == 4 : self.kin_inputs = Xsamp4
        elif self.win_size == 2 : self.kin_inputs = Xsamp2


        # self.vrate_0to3 = lasagne.layers.get_output(self.NN_XtoV_0to3, inputs = self.Xsamp[0:8])
        # self.vrate_rest = lasagne.layers.get_output(self.NN_XtoV, inputs = T.concatenate([self.Xsamp[:-8], self.Xsamp[1:-7], self.Xsamp[2:-6], self.Xsamp[3:-5], self.Xsamp[4:-4], self.Xsamp[5:-3], self.Xsamp[6:-2], self.Xsamp[7:-1], self.Xsamp[8:]], axis=1))
        self.vrate_0to3 = lasagne.layers.get_output(self.NN_XtoV_0to3, inputs = self.Xsamp[0:self.win_size-1])
        self.vrate_rest = lasagne.layers.get_output(self.NN_XtoV, inputs = self.kin_inputs)

        # self.vrate = lasagne.layers.get_output(self.NN_XtoV, inputs = self.Xsamp)
        self.vrate = T.concatenate([self.vrate_0to3, self.vrate_rest], axis=0)

    def getParams(self):
        return [self.QChol] + [self.Q0Chol] + [self.vRChol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY) + lasagne.layers.get_all_params(self.NN_XtoV) + lasagne.layers.get_all_params(self.NN_XtoV_0to3) + lasagne.layers.get_all_params(self.NN_dyn)

    def sampleX(self, _N):
        _x0 = np.asarray(self.x0.eval(), dtype=theano.config.floatX)
        _Q0Chol = np.asarray(self.Q0Chol.eval(), dtype=theano.config.floatX)
        _QChol = np.asarray(self.QChol.eval(), dtype=theano.config.floatX)
        _A = np.asarray(self.A.eval(), dtype=theano.config.floatX)

        norm_samp = np.random.randn(_N, self.xDim).astype(theano.config.floatX)
        x_vals = np.zeros([_N, self.xDim]).astype(theano.config.floatX)
        
        x_vals[0] = _x0 + np.dot(norm_samp[0],_Q0Chol.T)

        for ii in xrange(_N-1):
            # x_vals[ii+1] = x_vals[ii].dot(_A.T) + norm_samp[ii+1].dot(_QChol.T)
            x_vals[ii+1] = lasagne.layers.get_output(self.NN_dyn, inputs = x_vals[ii]).eval() + norm_samp[ii+1].dot(_QChol.T)

        return x_vals.astype(theano.config.floatX)

    def sampleY(self):
        ''' Return a symbolic sample from the generative model. '''
        return self.srng.poisson(lam = self.rate, size = self.rate.shape)

    def sampleV(self):
        return self.vrate+T.dot(self.srng.normal([self.Xsamp.shape[0],self.vDim]),T.diag(self.vRChol).T)

    def sampleXY(self,_N):
        ''' Return real-valued (numpy) samples from the generative model. '''
        X = self.sampleX(_N)

        Y = np.random.poisson(lam = self.rate.eval({self.Xsamp: X}))
        return [X.astype(theano.config.floatX),Y.astype(theano.config.floatX)]
    
    def evaluateLogDensity(self,X,Y,V):
        # This is the log density of the generative model (*not* negated)
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        # resX  = X[1:]-T.dot(X[:-1],self.A.T)
        resX  = X[1:] - lasagne.layers.get_output(self.NN_dyn, inputs = X[:(X.shape[0]-1)])
        resX0 = X[0]-self.x0
        Vpred = theano.clone(self.vrate,replace={self.Xsamp: X})
        resV = V - Vpred
        LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        #LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        PoisDensity = T.sum(Y * T.log(Ypred)  - Ypred - T.gammaln(Y + 1))
        KinDensity = -(0.5*T.dot(resV.T,resV)*T.diag(self.vRinv)).sum() + 0.5*(T.log(self.vRinv)).sum()*Y.shape[0] - 0.5*(self.vDim)*np.log(2*np.pi)*Y.shape[0]
        LogDensity = LatentDensity + PoisDensity + KinDensity
        return LogDensity
    
class PNNKinematicsWindowFD(LDS):
    '''
    Gaussian linear dynamical system with Poisson count observations. Inherits Gaussian
    linear dynamics sampling code from the LDS; implements a Poisson density evaluation
    for discrete (count) data.
    '''
    def __init__(self, GenerativeParams, xDim, yDim, A, QChol, Q0Chol, srng = None, nrng = None):
        # dimension of kinematics


        # The LDS class expects "RChol" for Gaussian observations - we just pass a dummy
        GenerativeParams['RChol'] = np.ones(1)
        super(PNNKinematicsWindowFD, self).__init__(GenerativeParams,xDim,yDim,A, QChol, Q0Chol,srng,nrng)
        self.vDim = 2;

        self.win_size = GenerativeParams['win_size']

        # Currently we emulate a PLDS by having an exponential output nonlinearity.
        # Next step will be to generalize this to more flexible output nonlinearities...
        if GenerativeParams['output_nlin'] == 'exponential':
            self.rate = T.exp(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'sigmoid':
            self.rate = T.nnet.sigmoid(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'softplus':
            self.rate = T.nnet.softplus(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        else:
            raise Exception('Unknown output nonlinearity specification!')
            
        if 'NN_XtoV_Params' in GenerativeParams:
            self.NN_XtoV = GenerativeParams['NN_XtoV_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_XtoV = lasagne.layers.DenseLayer(gen_nn, vDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())

        if 'NN_dyn_Params' in GenerativeParams:
            self.NN_dyn = GenerativeParams['NN_dyn_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_dyn = lasagne.layers.DenseLayer(gen_nn, xDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())

        if 'NN_XtoV_Params_0to3' in GenerativeParams:
            self.NN_XtoV_0to3 = GenerativeParams['NN_XtoV_Params_0to3']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_XtoV_0to3 = lasagne.layers.DenseLayer(gen_nn, vDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())

        if 'vRChol' in GenerativeParams:
            self.vRChol  = theano.shared(value=np.ndarray.flatten(GenerativeParams['vRChol'].astype(theano.config.floatX)), name='vRChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.vRChol  = theano.shared(value=np.random.randn(self.vDim).astype(theano.config.floatX)/10, name='vRChol' ,borrow=True)

        self.vRinv    = 1./(self.vRChol**2)
        
        # self.vrate_0to3 = lasagne.layers.get_output(self.NN_XtoV_0to3, inputs = self.Xsamp[0:4])
        # self.vrate_rest = lasagne.layers.get_output(self.NN_XtoV, inputs = T.concatenate([self.Xsamp[:-4], self.Xsamp[1:-3], self.Xsamp[2:-2], self.Xsamp[3:-1], self.Xsamp[4:]], axis=1))

        ## win_size = 14
        Xsamp14 = T.concatenate([self.Xsamp[:-13], self.Xsamp[1:-12], self.Xsamp[2:-11], self.Xsamp[3:-10], self.Xsamp[4:-9], self.Xsamp[5:-8], self.Xsamp[6:-7], self.Xsamp[7:-6], self.Xsamp[8:-5], self.Xsamp[9:-4], self.Xsamp[10:-3], self.Xsamp[11:-2], self.Xsamp[12:-1], self.Xsamp[13:]], axis=1)

        ## win_size = 12
        Xsamp12 = T.concatenate([self.Xsamp[:-11], self.Xsamp[1:-10], self.Xsamp[2:-9], self.Xsamp[3:-8], self.Xsamp[4:-7], self.Xsamp[5:-6], self.Xsamp[6:-5], self.Xsamp[7:-4], self.Xsamp[8:-3], self.Xsamp[9:-2], self.Xsamp[10:-1], self.Xsamp[11:]], axis=1)

        ## win_size = 10
        Xsamp10 = T.concatenate([self.Xsamp[:-9], self.Xsamp[1:-8], self.Xsamp[2:-7], self.Xsamp[3:-6], self.Xsamp[4:-5], self.Xsamp[5:-4], self.Xsamp[6:-3], self.Xsamp[7:-2], self.Xsamp[8:-1], self.Xsamp[9:]], axis=1)

        ## win_size = 8
        Xsamp8 = T.concatenate([self.Xsamp[:-7], self.Xsamp[1:-6], self.Xsamp[2:-5], self.Xsamp[3:-4], self.Xsamp[4:-3], self.Xsamp[5:-2], self.Xsamp[6:-1], self.Xsamp[7:]], axis=1)

        ## win_size = 6
        Xsamp6 = T.concatenate([self.Xsamp[:-5], self.Xsamp[1:-4], self.Xsamp[2:-3], self.Xsamp[3:-2], self.Xsamp[4:-1], self.Xsamp[5:]], axis=1)

        ## win_size = 4
        Xsamp4 = T.concatenate([self.Xsamp[:-3], self.Xsamp[1:-2], self.Xsamp[2:-1], self.Xsamp[3:]], axis=1)
        
        if self.win_size == 14 : self.kin_inputs = Xsamp14
        elif self.win_size == 10 : self.kin_inputs = Xsamp10
        elif self.win_size == 8 : self.kin_inputs = Xsamp8
        elif self.win_size == 6 : self.kin_inputs = Xsamp6
        elif self.win_size == 4 : self.kin_inputs = Xsamp4


        # self.vrate_0to3 = lasagne.layers.get_output(self.NN_XtoV_0to3, inputs = self.Xsamp[0:8])
        # self.vrate_rest = lasagne.layers.get_output(self.NN_XtoV, inputs = T.concatenate([self.Xsamp[:-8], self.Xsamp[1:-7], self.Xsamp[2:-6], self.Xsamp[3:-5], self.Xsamp[4:-4], self.Xsamp[5:-3], self.Xsamp[6:-2], self.Xsamp[7:-1], self.Xsamp[8:]], axis=1))
        self.vrate_0to3 = lasagne.layers.get_output(self.NN_XtoV_0to3, inputs = self.Xsamp[0:self.win_size-1])
        self.vrate_rest = lasagne.layers.get_output(self.NN_XtoV, inputs = self.kin_inputs)

        # self.vrate = lasagne.layers.get_output(self.NN_XtoV, inputs = self.Xsamp)
        self.vrate = T.concatenate([self.vrate_0to3, self.vrate_rest], axis=0)

    def getParams(self):
        return [self.QChol] + [self.Q0Chol] + [self.vRChol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY) + lasagne.layers.get_all_params(self.NN_XtoV) + lasagne.layers.get_all_params(self.NN_XtoV_0to3)

    def sampleX(self, _N):
        _x0 = np.asarray(self.x0.eval(), dtype=theano.config.floatX)
        _Q0Chol = np.asarray(self.Q0Chol.eval(), dtype=theano.config.floatX)
        _QChol = np.asarray(self.QChol.eval(), dtype=theano.config.floatX)
        _A = np.asarray(self.A.eval(), dtype=theano.config.floatX)

        norm_samp = np.random.randn(_N, self.xDim).astype(theano.config.floatX)
        x_vals = np.zeros([_N, self.xDim]).astype(theano.config.floatX)
        
        x_vals[0] = _x0 + np.dot(norm_samp[0],_Q0Chol.T)

        for ii in xrange(_N-1):
            # x_vals[ii+1] = x_vals[ii].dot(_A.T) + norm_samp[ii+1].dot(_QChol.T)
            x_vals[ii+1] = lasagne.layers.get_output(self.NN_dyn, inputs = x_vals[ii]).eval() + norm_samp[ii+1].dot(_QChol.T)

        return x_vals.astype(theano.config.floatX)

    def sampleY(self):
        ''' Return a symbolic sample from the generative model. '''
        return self.srng.poisson(lam = self.rate, size = self.rate.shape)

    def sampleV(self):
        return self.vrate+T.dot(self.srng.normal([self.Xsamp.shape[0],self.vDim]),T.diag(self.vRChol).T)

    def sampleXY(self,_N):
        ''' Return real-valued (numpy) samples from the generative model. '''
        X = self.sampleX(_N)

        Y = np.random.poisson(lam = self.rate.eval({self.Xsamp: X}))
        return [X.astype(theano.config.floatX),Y.astype(theano.config.floatX)]
    
    def evaluateLogDensity(self,X,Y,V):
        # This is the log density of the generative model (*not* negated)
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        # resX  = X[1:]-T.dot(X[:-1],self.A.T)
        resX  = X[1:] - lasagne.layers.get_output(self.NN_dyn, inputs = X[:(X.shape[0]-1)])
        resX0 = X[0]-self.x0
        Vpred = theano.clone(self.vrate,replace={self.Xsamp: X})
        resV = V - Vpred
        LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        #LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        PoisDensity = T.sum(Y * T.log(Ypred)  - Ypred - T.gammaln(Y + 1))
        KinDensity = -(0.5*T.dot(resV.T,resV)*T.diag(self.vRinv)).sum() + 0.5*(T.log(self.vRinv)).sum()*Y.shape[0] - 0.5*(self.vDim)*np.log(2*np.pi)*Y.shape[0]
        LogDensity = LatentDensity + PoisDensity + KinDensity
        return LogDensity