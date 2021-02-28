"""
    pyDeepDelta.py - A Python/TensorFlow module for predictive epistemic 
    uncertainty  quantification in deep learning classication models, as 
    described in the paper "Epistemic Uncertainty Quantification in Deep 
    Learning Classification by the Delta Method" found at: 
    https://arxiv.org/abs/1912.00832
     
    Copyright (c) 2018-2021 by Geir K. Nilsen (geir.kjetil.nilsen@gmail.com)
    and the University of Bergen.
 
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
 
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os
import tensorflow.compat.v1 as tf
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.disable_v2_behavior()
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from utils import *
import tqdm
import re
import pdb
from sklearn.decomposition import IncrementalPCA

class DeepDelta(object):
    """ Implements the Delta Method for Epistemic Uncertainty Quantification 
        in Deep Learning Classification as described in the paper found at
        https://arxiv.org/abs/1912.00832
       
        Public Attributes:
            params: List of model parameters (list of tensor(s))
            P: Total number of model parameters (int)
            layers: List of layer attributes (list of string(s))
            layers_params: List of model parameters for the layers specified by
                           the layers attribute (list of tensor(s))
            layers_P: Total number of model parameters for the layers specified by
                      the layers attribute (int)
            cost: Cost function output (tensor)
            reg_lambda: L2-regularization parameter (tensor)
            reg_lambda_val: L2-regularization parameter value (float)
            yhat: Model function output (tensor)
            X: Model data input (tensor)
            y: Model data output (tensor)
            X_train: Training set input (ndarray)
            y_train: Training set output (ndarray)
            batch_size_H: Batchsize used when computing exact Hessian vector 
            products.
            batch_size_G: Batchsize used when computing OPG approximation
            batch_size_F: Batchsize used when computing the sensitivity matrix
            K: Number of principal Hessian eigenpairs used in the covariance 
            approximation (int)
            tfsession: TensorFlow session (object)
            L: The total number of layers in the model.
            TL: The total number of model function outputs/classes.
    
    """

    def __init__(self, params, cost, reg_lambda, reg_lambda_val, yhat, X, y, 
                 X_train, y_train, batch_size_H, K, tfsession, layers=None, rseed=None,
                 batch_size_G=None, model_fun=None, cost_fun=None, cost_fun_data=None, 
                 dropout_rate=None, dropout_rate_val=None, batch_size_F=None, logits=False):

        """
        Args:
            params: List of model parameters (list of tensor(s))
            cost: Cost function output (tensor)
            reg_lambda: L2-regularization parameter (tensor)
            reg_lambda_val: L2-regularization parameter value (float)
            yhat: Model function output (tensor)
            X: Model input data (tensor)
            y: Model output data (tensor)
            X_train: Model input training data (ndarray)
            y_train: Model output training data (ndarray)
            batch_size_H: Batchsize used when computing exact Hessian vector
                          products (int).
            K: Number of Hessian eigenpairs to calculate and base the covariance
               approximation on (int)
            tfsession: TensorFlow session (object)
            layers: List of layer attributes (list of string(s)). Used to 
                    specify which 'part(s)' of the model architecture to focus
                    on in the uncertainty approximations. Defaults to the full 
                    model. Format: 'lw', 'lb' or 'lwb' where l is the layer 
                    number ranging from l = 2 to L, where L is the total number 
                    of layers, and where 'w' denotes to include weights and 'b' 
                    denotes to include biases. By convention, layer l = 1 is 
                    defined as the input layer which has no weights and biases.
                    Example: layers=['2w', '4wb', '6b']
                    -> Will result in exclusion of layers 3 and 5, and only 
                    include weights from layer 2, both weights and biases from 
                    layer 4, and only biases from layer 6.
            rseed: random seed to ensure repeatability [used when generating the
                   start vector for the Lanczos iteration] (int).
            batch_size_G: Batchsize used when computing OPG approximation (int).
            model_fun: Model function (function).
            cost_fun: Cost function [with L2 reg] (function).
            cost_fun_data: Cost function (without L2 reg).
            dropout_rate: Dropout regularization parameter (tensor).
            dropout_rate_val: Dropout regularization parameter value (float).
            batch_size_F: Batchsize used when computing the sensitivity matrix 
                          (int).
            logits: If True, sensitivity matrix will be computed in logits 
                    (bool).
          
        """
        
        # Attributes
        self.params = params
        self.P = self.flatten(self.params).get_shape().as_list()[0]
        self.cost = cost
        self.reg_lambda = reg_lambda
        self.reg_lambda_val = reg_lambda_val
        self.dropout_rate = dropout_rate
        self.dropout_rate_val = dropout_rate_val
        self.yhat = yhat
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = None
        self.y_train = y_train
        self.batch_size_H = batch_size_H
        if batch_size_G is None:
            self.batch_size_G = 100    
        else:
            self.batch_size_G = batch_size_G
        self.N_train = X_train.shape[0]
        if self.N_train % self.batch_size_H != 0:
            raise Exception('N_train must be divisible by batch_size_H!')
        self.tfsession = tfsession
        self.L = int(len(self.params)/2)+1
        if layers is None:
            self.layers = ['%dwb' % (l+2) for l in range(self.L-1)]
        else:
            self.layers = layers
        self.layers_params = [y for x in [self._parse_params(self.params, 
                                                           l)
                                          for l in range(len(self.layers))] \
                              for y in x]
        self.layers_P = self.flatten(self.layers_params).get_shape().as_list()[0]
        self.K = K
        self.TL = self.yhat.shape.as_list()[1]
               
        self.output_gradient_op = None
        self._logits = logits
        self.batched_output_gradient_op = None
        if batch_size_F is None:
            self.batch_size_F = 100
        else:
            self.batch_size_F = batch_size_F
        self._batched_var_H_op = None
        self._batched_var_G_op = None
        self._var_G_op = None
        self._var_G_err_op = None
        self._sens_op = None
        self._batched_sens_op = None
        self._batched_var_H_err_op = None
        self._batched_var_G_err_op = None
        self._batched_var_Sandwich_err_op = None
        self._exact_var_H_op = None
        self._exact_var_Sandwich_op = None
        self._Hv_op = None
        self._Gv_op = None
        self._F_op = None
        self._batched_F_op = None
        self._G_op = None
        self._H_op = None
        self._C_op = None
        self._batched_var_Sandwich_op = None
        self._model_fun = model_fun
        self._cost_fun = cost_fun
        self._cost_fun_data = cost_fun_data
        self._Q_H_init = None
        self._Q_H = None
        self._Lambda_H = None
        self._Q_G_init = None
        self._Q_G = None
        self._Lambda_G = None
        self._H = None
        self._G = None
        self._b = 0
        self._H_Kinds = tf.placeholder(tf.int32, shape=(None))
        self._G_Kinds = tf.placeholder(tf.int32, shape=(None))
        self._lambda_H_gap_inv = tf.placeholder(tf.float32)
        self._lambda_H_gap_err_inv = tf.placeholder(tf.float32)
        self._lambda_G_gap_inv = tf.placeholder(tf.float32)
        self._lambda_G_gap = tf.placeholder(tf.float32)
        self._lambda_G_gap_err_inv = tf.placeholder(tf.float32)
        self._plambda_H = tf.placeholder(tf.float32)
        self._plambda_G = tf.placeholder(tf.float32)
        self._v = None
        self._eigsh_tqdm = None
        self._q = tf.placeholder(tf.int32, shape=())
        if rseed is None:
            self._rseed = 2021
        else:
            self._rseed = rseed
        
        print('DeepDelta initialized.\nP = %d, K = %d, N = %d' % (self.layers_P, 
                                                                  self.K, 
                                                                  self.N_train))
    
    # Functions
    
    def _init_Kinds(self, K=None, which=None):
        # Generate and return array representing the indices of the relevant 
        # eigenpairs from Q and Lambda.
        
        if K is None:
            K = self.K
               
        Kinds = np.arange(0, K)

        if which is None:
            return Kinds
        elif which is 'plambda': # Most similar to lambda from positive side
             Kinds = np.expand_dims(Kinds[-1], axis=0)
        else:
            raise NotImplementedError()
    
        return Kinds
       
        
    def _parse_params(self, params, layer):
        # Extract relevant tensors from the list params for the layer
        try:
            dl = int(re.sub(r'\D', "", self.layers[layer]))
            dm = re.sub(r'\d', "", self.layers[layer])
        
            if (dl-2)*2 < 0 or (dl-2)*2+1 < 0 or (dl-2)*2 > len(params) \
                            or (dl-2)*2+1 > len(params):
                raise Exception('Invalid layer number!')
        
            if dm == 'wb':
                return [params[(dl-2)*2]]+[params[(dl-2)*2+1]]
            elif dm == 'w':
                return [params[(dl-2)*2]]
            elif dm == 'b':
                return [params[(dl-2)*2+1]]
            else:
                raise Exception('Syntax error while parsing layers attribute!')
        except:
            raise Exception('Syntax error while parsing layers attribute!')
        return
    
    def _progress(self, iterable, display=True):
        # Tqdm wrapper
        if display is True:
            return tqdm.tqdm(iterable, position=0, leave=True)
        else:
            return iterable
            
          
    def _Hv(self, v):
        # Callback function for LinearOperator used with eigsh(). Used to evaluate
        # a Hessian vector product op for the training set. Returns evaluated product 
        # of the Hessian multiplied with the vector v.
        B = int(self.N_train/self.batch_size_H)
        Hv = np.zeros((self.layers_P))
        Bs = self.batch_size_H
        for b in range(B):
            Hv = Hv + self.tfsession.run(self._Hv_op, 
                                       feed_dict={self.X:self.X_train[b*Bs:(b+1)*Bs], 
                                                  self.y:self.y_train[b*Bs:(b+1)*Bs], 
                                                  self.reg_lambda: self.reg_lambda_val, 
                                                  self.dropout_rate: self.dropout_rate_val,
                                                  self._v:np.squeeze(v)})
        Hv = Hv / B
        self._b = self._b + 1
        self._eigsh_tqdm.update()
        return Hv


    def _Gv(self, v):
        # Callback function for LinearOperator used with eigsh(). Used to evaluate
        # a OPG vector product op for the training set. Returns evaluated product 
        # of the OPG multiplied with the vector v.
        B = int(self.N_train/self.batch_size_G)
        Gv = np.zeros((self.layers_P))
        Bs = self.batch_size_G
        for b in range(B):
            Gv = Gv + self.tfsession.run(self._Gv_op, 
                                       feed_dict={self.X:self.X_train[b*Bs:(b+1)*Bs], 
                                                  self.y:self.y_train[b*Bs:(b+1)*Bs], 
                                                  self.reg_lambda: self.reg_lambda_val, 
                                                  self.dropout_rate: self.dropout_rate_val,
                                                  self._v:np.squeeze(v)})
        Gv = Gv / B
        self._b = self._b + 1
        self._eigsh_tqdm.update()
        return Gv
          

    def flatten(self, params):
        """
        Implements an op for flattening of the list of tensor(s) params into a
        tensor of shape (P).
        
        Args:
            params: A list of tensors
        
        Returns:
            A tensor of shape (P)
        """
        return tf.concat([tf.reshape(_params, [-1]) \
                          for _params in params], axis=0)
       

    def unflatten(self, params_flat):
        """
        Implements an op for unflattening of the (P)-shaped tensor params_flat 
        into a list of tensor(s), e.g. inverse of flatten().
        
        Args:
            params_flat: tensor of shape (P)
        
        Returns:
            A list of tensors
        """
        params = []
        for p in range(len(self.params)):
            if p == 0:
                start = 0
            else:
                start = start + np.prod(self.params[p-1].shape.as_list())
            stop = start + np.prod(self.params[p].shape.as_list())
            params.append(tf.reshape(params_flat[start:stop], 
                                     self.params[p].shape))
        return params

    def get_output_gradient_op(self, output):
        """ 
        Implements an op for the gradient of the given output of the 
        model function (wrt model parameters) 
             
        Args:
            output: output number to differentiate
        
        Returns:
            An op for the gradient of the given output of the model function 
            (wrt model parameters) (tensor of shape (P))
        """
        if self.output_gradient_op is None:
            self.output_gradient_op = self.flatten(tf.gradients(self.yhat[:,output], 
                                                                 self.layers_params))
        return self.output_gradient_op


    def get_batched_output_gradient_op(self, output):
        """ 
        Implements a batched op for the gradient of the given output of the 
        model function (wrt model parameters) 
             
        Args:
            output: output number to differentiate
        
        Returns:
            A batched op for the gradient of the given output of the model 
            function (wrt model parameters). Faster than the unbatched version.
        """
                       
        ex_params = [[tf.identity(_params) \
                      for _params in self.params] \
                     for ex in range(self.batch_size_F)]      
        
        ex_X = tf.split(self.X, self.batch_size_F)
        if self._logits is False:
            # NB: Loss of generality: we assume here that the output layer has 
            #     softmax activation function.
            ex_yhat = [tf.nn.softmax(self._model_fun(_X, _params)) \
                       for _X, _params in zip(ex_X, ex_params)]
        else:
            ex_yhat = [self._model_fun(_X, _params) \
                       for _X, _params in zip(ex_X, ex_params)]
        self.batched_output_gradient_op = tf.stack([self.flatten(tf.gradients(ex_yhat[ex][0, output], 
                                                       [y for x in [self._parse_params(ex_params[ex], l)
                                                                    for l in range(len(self.layers))]
                                                        for y in x]))
                                                     for ex in range(self.batch_size_F)])
        return self.batched_output_gradient_op


    def get_F_op(self):
        """
        Implements an op for the Jacobian (wrt model parameters) of the model 
        function.
        
        Args:
            None
            
        Returns:
            An op for the Jacobian of the model function (tensor of shape (TL, P))
        """
        if self._F_op is None:
            self._F_op = tf.map_fn(self.get_output_gradient_op, tf.range(0, self.TL), 
                                   dtype='float32')
        return self._F_op


    def get_batched_F_op(self):
        """
        Imeplements a batched op for the Jacobian (wrt model parameters) of the 
        model function.

        Args:
            None

        Returns:
            A batched op for the Jacobian of the model function 
            (tensor of shape (batch_size_F, TL, P))

        """

        self._batched_F_op = tf.cast(tf.transpose(tf.map_fn(self.get_batched_output_gradient_op,
                                                            tf.range(0, self.TL),
                                                            dtype='float32'), 
                                                  (1,0,2)), 
                                     'float32')
        return self._batched_F_op

    
    def eval_F_op(self, X, progress=True):
        """
        Evaluates op for the Jacobian (wrt model parameters) of the model 
        function.
        
        Args:
            X: ndarray of input data
            progress (optional): Boolean, if True display progress information,
                                 defaults to True.
            
        Returns:
            ndarray of shape (X.shape[0], TL, P) representing the Jacobian of 
            the model function
        """
        N = np.shape(X)[0]
        if self._F_op is None:
            self._F_op = self.get_F_op()
                
        return np.array([self.tfsession.run(self._F_op, 
                                            feed_dict={self.X:[X[n]]})                                                       
                         for n in self._progress(range(N), progress)])

    
    def eval_batched_F_op(self, X, progress=True):
        """
        Evaluates batched op for the Jacobian (wrt model parameters) of the model 
        function. Batched version is generally faster than the unbatched 
        version.
        
        Args:
            X: ndarray of input data
            progress (optional): Boolean, if True display progress information,
                                 defaults to True.
            
        Returns:
            ndarray of shape (X.shape[0], TL, P) representing the Jacobian of the 
            model function
        """
        N = np.shape(X)[0]
        B = int(N/self.batch_size_F)
        if self._batched_F_op is None:
            self._batched_F_op = self.get_batched_F_op()
                
        return np.concatenate([self.tfsession.run(self._batched_F_op, 
                                            feed_dict={self.X:X[b*self.batch_size_F:(b+1)*self.batch_size_F]})  
                         for b in self._progress(range(B), progress)])


    def get_sens_op(self):
        """
        Get op for output sensitivity, e.g. treat covariance matrix as the identity 
        matrix and return diag(F@I@F.T)
        Args:
            None
            
        Returns:
            An op for the sensitivity of the model output (tensor of shape (TL))        
        """
        if self._F_op is None:
            self._F_op = self.get_F_op()
        if self._sens_op is None:
            self._sens_op = tf.reduce_sum(self._F_op**2, axis=1)

        return self._sens_op


    def get_batched_sens_op(self):
        """
        Get batched op for output sensitivity, e.g. treat covariance matrix as 
        the identity matrix and return diag(F@I@F.T)
        Args:
            None
            
        Returns:
            An op for the output sensitivity, tensor of shape (batch_size_F, TL)            
        """
        if self._batched_F_op is None:
            self._batched_F_op = self.get_batched_F_op()
        if self._batched_sens_op is None:
            self._batched_sens_op = [tf.reduce_sum(self._batched_F_op[b]**2, axis=1) 
                                     for b in range(self.batch_size_F)]

        return self._batched_sens_op

    
    def eval_sens_op(self, X, progress=True):
        """
        Eval op for output sensitivity, diag(F@I@F.T).
        Args:
            X: ndarray of input data
            progress (optional): Boolean, if True display progress information,
                                 defaults to True.
           
        Returns:
            ndarray of shape (X.shape[0], TL)
           
        """
        N = np.shape(X)[0]
                            
        if self._sens_op is None:
            self._sens_op = self.get_sens_op()
                
        return np.array([self.tfsession.run(self._sens_op, 
                                                feed_dict={self.X:[X[n]]})
                             for n in self._progress(range(N), progress)])


    def eval_batched_sens_op(self, X, progress=True):
        """
        Eval batched op for output sensitivity. Generally faster than unbatched 
        version.
        Args:
            X: ndarray of input data
            progress (optional): Boolean, if True display progress information,
                                 defaults to True.
           
        Returns:
            ndarray of shape (X.shape[0], TL)
           
        """
        N = np.shape(X)[0]
        B = int(N/self.batch_size_F)
                            
        if self._batched_sens_op is None:
            self._batched_sens_op = self.get_batched_sens_op()
                
        return np.concatenate([self.tfsession.run(self._batched_sens_op, 
                                            feed_dict={self.X:X[b*self.batch_size_F:(b+1)*self.batch_size_F]})  
                         for b in self._progress(range(B), progress)])


    def get_Hv_op(self, v):
        """ 
        Implements an exact Hessian vector product op defined as the 
        matrix multiplication of the Hessian matrix (of the cost function) H
        with the vector v. Note: This implementation exploits the fact that the
        outermost differentiation (e.g. tf.gradients) acts on a vector function 
        (e.g. gradient) rather than a scalar function (as is the case in the 
        innermost differentiation). Thus, the outermost differentiation will 
        return the _sums_ of the derivatives of the innermost derivatives. 
        Since the innermost derivatives are multiplied element-wise by v, 
        the result will evaluate to the full matrix of second order partial 
        derivates of the cost function implicitely multiplied by the vector v.
    
        Args:      
            v: Vector to multiply by the Hessian (tensor)
        
        Returns:
            Hv_op: Hessian vector product op (tensor of the same shape as v).
        """
        if self._Hv_op is None:
            self._Hv_op = self.flatten(tf.gradients(tf.math.multiply(self.flatten(tf.gradients(self.cost, 
                                                                                               self.layers_params)), 
                                                                     tf.stop_gradient(v)), 
                                                    self.layers_params))
        return self._Hv_op

               
    def init_eig_H_op(self, Lambda, Q=None):
        """
        Convert and initialize pre-calculated Hessian eigenpairs as tensors.
        
        Args:
            Lambda: array of shape (K) representing the K algebraically largest Hessian 
                    eigenvalues.
            Q: ndarray of shape (P, K) representing the corresponding Hessian eigenvectors.
        
        Returns:
            None
        """
        
        self._Lambda_H = tf.constant(Lambda, dtype='float32')
        if Q is not None:
            self._Q_H_init = tf.placeholder(tf.float32, shape=(self.layers_P, self.K))
            self._Q_H = tf.Variable(self._Q_H_init, dtype='float32')
            self.tfsession.run(self._Q_H.initializer, feed_dict={self._Q_H_init: Q})
        return


    def init_eig_G_op(self, Lambda, Q=None):
        """
        Convert and initialize pre-calculated OPG eigenpairs as tensors.
        
        Args:
            Lambda: array of shape (K) representing the K algebraically largest OPG
                    eigenvalues.
            Q: ndarray of shape (P, K) representing the corresponding OPG eigenvectors.
        
        Returns:
            None
        """
        
        self._Lambda_G = tf.constant(Lambda, dtype='float32')
        if Q is not None:
            self._Q_G_init = tf.placeholder(tf.float32, shape=(self.layers_P, self.K))
            self._Q_G = tf.Variable(self._Q_G_init, dtype='float32')
            self.tfsession.run(self._Q_G.initializer, feed_dict={self._Q_G_init: Q})
        return
      
    
    def compute_eig_H(self, progress=True):
        """
        Calculate and return the K algebraically largest Hessian eigenvalues 
        (and corresponding eigenvectors).
        The procedure is based on exact Hessian vector products and the Lanczos 
        iteration.
        
        Args:
            progress (optional): Boolean, if True display progress information,
                                 defaults to True.
        
        Returns:
            Lambda: tensor of shape (K) representing the K algebraically largest 
                    Hessian eigenvalues.
            Q: ndtensor of shape (P, K) representing the corresponding K 
               eigenvectors.
            its: The total number of Lanczos-Pearlmutter steps that was carried
                 out before convergence.
        """
        if self._v is None:
            self._v = tf.placeholder(shape=(self.layers_P,), dtype='float32')
        if self._Hv_op is None:
            self._Hv_op = self.get_Hv_op(self._v)
            
        print('Starting Lanczos Iteration...', flush=True)
        H = LinearOperator((self.layers_P, self.layers_P), matvec=self._Hv, 
                           dtype='float32')
        self._b = 0
        if progress is True:
            self._eigsh_tqdm=tqdm.tqdm(self._b, position=0, leave=True)
        if self._rseed is not None:
            # Guarantees repeatability
            np.random.seed(self._rseed)
        v0 = np.random.normal(size=(self.layers_P))
        Lambda,Q = eigsh(H, k=self.K, v0=v0, which='LA')
        print('\nLanczos finished in %d iterations.' % self._b, flush=True)
        inds=np.flip(np.argsort(Lambda))
        Lambda = Lambda[inds]
        Q = Q[:,inds]

        return Lambda, Q, self._b

    def get_G_op(self):
        """ 
        Implements an op for the full OPG approximation by a per-example 
        cost Jacobian matrix product
     
        Args:
            None
        
        Returns:
            G_op: OPG approximation op (tensor of shape (P, P))
        """
                       
        ex_params = [[tf.identity(_params) \
                      for _params in self.params] \
                     for ex in range(self.batch_size_G)]      
        
        ex_X = tf.split(self.X, self.batch_size_G)
        ex_y = tf.split(self.y, self.batch_size_G)    
        ex_yhat_logits = [self._model_fun(_X, _params) \
                          for _X, _params in zip(ex_X, 
                                                 ex_params)]
        ex_cost = [self._cost_fun(_y, _yhat_logits, _params) \
                   for _y, _yhat_logits, _params in zip(ex_y, 
                                                        ex_yhat_logits,
                                                        ex_params)]
        ex_grads = tf.stack([self.flatten(tf.gradients(ex_cost[ex], 
                                                       [y for x in [self._parse_params(ex_params[ex], 
                                                           l)
                                          for l in range(len(self.layers))] \
                              for y in x]))
                             for ex in range(self.batch_size_G)])
        G_op = tf.matmul(tf.transpose(ex_grads), 
                         ex_grads) / self.batch_size_G
        return G_op


    def get_Gv_op(self, v):
        """ 
        Implements an op for a OPG approximation vector product.
     
        Args:
            v: vector to multiply by the OPG approxmation
        
        Returns:
            Gv_op: OPG approximation vector product op (tensor of shape (P))
        """
                       
        ex_params = [[tf.identity(_params) \
                      for _params in self.params] \
                     for ex in range(self.batch_size_G)]      
        
        ex_X = tf.split(self.X, self.batch_size_G)
        ex_y = tf.split(self.y, self.batch_size_G)    
        ex_yhat_logits = [self._model_fun(_X, _params) \
                          for _X, _params in zip(ex_X, 
                                                 ex_params)]
        ex_cost = [self._cost_fun(_y, _yhat_logits, _params) \
                   for _y, _yhat_logits, _params in zip(ex_y, 
                                                        ex_yhat_logits,
                                                        ex_params)]
        ex_grads = tf.stack([self.flatten(tf.gradients(ex_cost[ex], 
                                                       [y for x in [self._parse_params(ex_params[ex], l)
                                                                    for l in range(len(self.layers))]
                                                        for y in x]))
                             for ex in range(self.batch_size_G)])
        Gv_op = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(ex_grads), 
                         ex_grads), tf.expand_dims(v, axis=1))) / self.batch_size_G
        return Gv_op

    
    def eval_G_op(self, progress=True):
        """
        Evaluates op for the OPG approximation.
        
        Args:
            progress (optional): Boolean, if True display progress information,
                                 defaults to True.
            
        Returns:
            ndarray of shape (P, P) representing the OPG approximation.
        """
        if self._G_op is None:
            self._G_op = self.get_G_op()
                
        G = np.zeros((self.layers_P, self.layers_P), dtype='float32')
        B = int(self.N_train/self.batch_size_G)
        for b in self._progress(range(B), progress):
            G = G + self.tfsession.run(self._G_op, feed_dict={self.X:self.X_train[b*self.batch_size_G: \
                                                              (b+1)*self.batch_size_G], 
                                                              self.y:self.y_train[b*self.batch_size_G: \
                                                              (b+1)*self.batch_size_G],
                                                              self.reg_lambda:self.reg_lambda_val})
        return G / B
    


    def get_C_op(self):
        """ 
        Implements an op for the Jacobian of the data-dependent part of the cost
        function. 
        Args:
            None
        
        Returns:
            ndarray of shape (batch_size_G, P)
        """
                       
        ex_params = [[tf.identity(_params) \
                      for _params in self.params] \
                     for ex in range(self.batch_size_G)]      
        
        ex_X = tf.split(self.X, self.batch_size_G)
        ex_y = tf.split(self.y, self.batch_size_G)    
        ex_yhat_logits = [self._model_fun(_X, _params) \
                          for _X, _params in zip(ex_X, 
                                                 ex_params)]
        ex_cost = [self._cost_fun_data(_y, _yhat_logits) \
                   for _y, _yhat_logits in zip(ex_y, 
                                               ex_yhat_logits)]
        ex_grads = tf.stack([self.flatten(tf.gradients(ex_cost[ex], 
                                                       [y for x in [self._parse_params(ex_params[ex], l)

                                                                    for l in range(len(self.layers))]
                                                        for y in x]))
                             for ex in range(self.batch_size_G)])
        self._C_op = ex_grads / self.batch_size_G
        return self._C_op


    def compute_eig_G(self, progress=True):
        """
        Calculate and return the K algebraically largest OPG approximation 
        eigenvalues (and corresponding eigenvectors).
        The procedure is based on incremental singular value decompositions of 
        the Jacobian matrix of the data-dependent part of the cost function.
        
        Args:
            progress (optional): Boolean, if True display progress information,
                                 defaults to True.
        
        Returns:
            Lambda: tensor of shape (K) representing the K algebraically largest
                    OPG eigenvalues.
            Q: ndtensor of shape (P, K) representing the corresponding K 
               eigenvectors.
        """

        N = int(np.ceil(self.K / self.batch_size_G))

        if self.N_train % N != 0:
            raise Exception('N_train must be divisible by K/batch_size_G!')

        ipca = IncrementalPCA(n_components=self.K, batch_size=self.batch_size_G*N, 
                              copy=False)

        if self._C_op is None:
            self._C_op = self.get_C_op()
        
        print('Starting Incremental SVD...', flush=True)
        
        
        C = np.zeros((self.batch_size_G*N, self.layers_P), dtype='float32')
        B = int(self.N_train/self.batch_size_G)
        for b in self._progress(range(B), progress):
            C[self.batch_size_G*(b%N):self.batch_size_G*(b%N+1),:] = self.tfsession.run(self._C_op, 
                                                                                        feed_dict={self.X:self.X_train[b*self.batch_size_G: \
                                                                                                                       (b+1)*self.batch_size_G], 
                                                                                                   self.y:self.y_train[b*self.batch_size_G: \
                                                                                                                       (b+1)*self.batch_size_G]})
            if (b+1) % N == 0 and b != 0:
                ipca.partial_fit(C)
                
        return np.float32(ipca.singular_values_**2 / (self.N_train/(self.batch_size_G)**2)) \
               + self.reg_lambda_val, np.float32(ipca.components_.T)
                                                              

    def get_H_op(self):
        """ 
        Implements a full Hessian matrix op by forming p Hessian vector 
        products using HessianEstimator.get_Hv_op(v) for all v's in R^P
        
        Args:
            None
        
        Returns:
            H_op: Hessian matrix op (tensor of shape (P, P))
        """
        H_op = tf.map_fn(self.get_Hv_op, tf.eye(self.layers_P, 
                                                self.layers_P), 
                         dtype='float32')
        return H_op

    
    def eval_H_op(self, progress=True):
        """
        Evaluates op for the full Hessian matrix.
        
        Args:
            progress (optional): Boolean, if True display progress information,
                                 defaults to True.
            
        Returns:
            ndarray of shape (P, P) representing the Hessian
        """
        if self._H_op is None:
            self._H_op = self.get_H_op()
        
        if self._H is None:
            self._H = np.zeros((self.layers_P, self.layers_P), dtype='float32')
        B = int(self.N_train/self.batch_size_H)
        for b in self._progress(range(B), progress):
            self._H = self._H + self.tfsession.run(self._H_op, 
                                                   feed_dict={self.X:self.X_train[b*self.batch_size_H: \
                                                                                  (b+1)*self.batch_size_H], 
                                                              self.y:self.y_train[b*self.batch_size_H: \
                                                                                  (b+1)*self.batch_size_H],
                                                              self.reg_lambda:self.reg_lambda_val})
        self._H = self._H / B
        return self._H

    
    def eval_exact_H_var_op(self, X, progress=True):
        """
        Evaluates the predictive variance of X using the full Hessian estimator 
        (no eigen approximation).

        Args:
            X: Input data
            progress (optional): Boolean, if True display progress information, 
                                 defaults to True.

        Returns:
            ndarray of shape (X.shape[0], TL) representing the predictive 
            variance of X.

        """
        N = X.shape[0]
        if self._F_op is None:
            self._F_op = self.get_F_op()

        if self._H is None:
            self._H = self.eval_H_op(progress=True)
                
        self._exact_H_var_op = tf.diag_part(tf.matmul(tf.matmul(self._F_op, tf.linalg.inv(self._H)), 
                                                      tf.transpose(self._F_op))) / self.N_train
        
        return np.array([self.tfsession.run(self._exact_H_var_op, 
                                            feed_dict={self.X:[X[n]],
                                                       self.reg_lambda:self.reg_lambda_val})
                         for n in self._progress(range(N), progress)])


    def eval_exact_Sandwich_var_op(self, X, progress=True):
        """
        Evaluates the predictive variance of X using the full Sandwich estimator 
        (no eigen approximation).

        Args:
            X: Input data
            progress (optional): Boolean, if True display progress information, 
                                 defaults to True.

        Returns:
            ndarray of shape (X.shape[0], TL) representing the predictive
            variance of X.

        """
        N = X.shape[0]
        if self._F_op is None:
            self._F_op = self.get_F_op()

        if self._H is None:
            self._H = self.eval_H_op(progress=True)
        if self._G is None:
            self._G = self.eval_G_op(progress=True)
        Cov = tf.linalg.inv(self._H)@(self._G + self.reg_lambda*tf.eye(self.layers_P, self.layers_P))@tf.linalg.inv(self._H)
        
        self._exact_Sandwich_var_op = tf.diag_part(tf.matmul(tf.matmul(self._F_op, Cov), 
                                                             tf.transpose(self._F_op))) / self.N_train
                
        return np.array([self.tfsession.run(self._exact_Sandwich_var_op, 
                                            feed_dict={self.X:[X[n]],
                                                       self.reg_lambda:self.reg_lambda_val})
                         for n in self._progress(range(N), progress)])


    def eval_exact_G_var_op(self, X, progress=True):
        """
        Evaluates the predictive variance of X using the full OPG estimator 
        (no eigen approximation).

        Args:
            X: Input data
            progress (optional): Boolean, if True display progress information, 
                                 defaults to True.

        Returns:
            ndarray of shape (X.shape[0], TL) representing the predictive 
            variance of X.

        """
        N = X.shape[0]
        if self._F_op is None:
            self._F_op = self.get_F_op()

        if self._H is None:
            self._H = self.eval_H_op(progress=True)
        if self._G is None:
            self._G = self.eval_G_op(progress=True)
        Cov = tf.linalg.inv(self._G + self.reg_lambda*tf.eye(self.layers_P, self.layers_P))
        
        exact_G_var_op = tf.diag_part(tf.matmul(tf.matmul(self._F_op, Cov), 
                                                tf.transpose(self._F_op))) / self.N_train
        
        
        return np.array([self.tfsession.run(exact_G_var_op, 
                                            feed_dict={self.X:[X[n]],
                                                       self.reg_lambda:self.reg_lambda_val})
                         for n in self._progress(range(N), progress)])
    

    def get_Q_H(self):
        """
        Return tensor with Hessian eigenvectors.
        
        Args:
            None
            
        Returns:
            ndtensor of shape (P, K)             
        """
        return self._Q_H
    

    def eval_Q_H(self, K=None, which=None):
        """
        Return evaluated Hessian eigenvectors.
        
        Args:
            K (optional): int, the number (<=K) of eigenvectors to extract. 
                          Defaults to K.
            which (optional): 'plambda' - the eigenvector corresponding to the 
                              algebraically smallest Hessian eigenvalue.

        Returns:
            ndarray of shape (P, K) or (P) 
        """
        
        Kinds = self._init_Kinds(K, which)
        return self.tfsession.run(tf.gather(self._Q_H, self._H_Kinds, axis=1), 
                                  feed_dict={self._H_Kinds:Kinds}) 
    

    def get_Lambda_H(self):
        """
        Return tensor with Hessian eigenvalues
        
        Args:
            None
            
        Returns:
            tensor of shape (K)
        """
        return self._Lambda_H
    

    def eval_Lambda_H(self, K=None, which=None):
        """
        Return evaluated Hessian eigenvalues.
        
        Args:
            K (optional): int, the number (<=K) of eigenvalues to extract. 
                          Defaults to K.
            which (optional): 'plambda' - the algebraically smallest Hessian 
                              eigenvalue.
                               
        Returns:
            array of shape (K) or () 
        """
        Kinds = self._init_Kinds(K, which)
        return self.tfsession.run(tf.gather(self._Lambda_H, self._H_Kinds, axis=0), 
                                  feed_dict={self._H_Kinds:Kinds}) 
    

    def get_Q_G(self):
        """
        Return tensor with OPG eigenvectors.
        
        Args:
            None
            
        Returns:
            ndtensor of shape (P, K)             
        """
        return self._Q_G
    

    def eval_Q_G(self, K=None, which=None):
        """
        Return evaluated OPG eigenvectors.
        
        Args:
            K (optional): int, the number (<=K) of eigenvectors to extract. 
                          Defaults to K.
            which (optional): 'plambda' - the eigenvector corresponding to the 
                              algebraically smallest OPG eigenvalue.

        Returns:
            ndarray of shape (P, K) or (P) 
        """
        
        Kinds = self._init_Kinds(K, which)
        return self.tfsession.run(tf.gather(self._Q_G, self._G_Kinds, axis=1), 
                                  feed_dict={self._G_Kinds:Kinds}) 
    

    def get_Lambda_G(self):
        """
        Return tensor with OPG eigenvalues
        
        Args:
            None
            
        Returns:
            tensor of shape (K)
        """
        return self._Lambda_G
    

    def eval_Lambda_G(self, K=None, which=None):
        """
        Return evaluated OPG eigenvalues.
        
        Args:
            K (optional): int, the number (<=K) of eigenvalues to extract. 
                          Defaults to K.
            which (optional): 'plambda' - the algebraically smallest OPG 
                              eigenvalue.

        Returns:
            array of shape (K) or () 
        """
        Kinds = self._init_Kinds(K, which)
        return self.tfsession.run(tf.gather(self._Lambda_G, self._G_Kinds, axis=0), 
                                  feed_dict={self._G_Kinds:Kinds}) 


    def get_batched_var_H_op(self):
        """
        Implements an op for the predictive variance using the approximate 
        Hessian estimator. [Equation (19) using equation (11) in the paper].
        
        Args:
            None
            
        Returns:
            tensor of shape (batch_size_F, TL)            
        """
        if self._Q_H is None or self._Lambda_H is None:
            self.init_eig_H_op(*self.compute_eig_H())
        if self._batched_F_op is None:
            self._batched_F_op = self.get_batched_F_op()
        if self._batched_var_H_op is None:
            self._batched_var_H_op = [(tf.reduce_sum(tf.matmul(self._batched_F_op[b], 
                                                               tf.gather(self._Q_H, 
                                                                         self._H_Kinds, 
                                                                         axis=1))**2 \
                                                     / tf.gather(self._Lambda_H,
                                                                 self._H_Kinds,
                                                                 axis=0),
                                                     axis=1) \
                                       + self._lambda_H_gap_inv*(tf.reduce_sum(self._batched_F_op[b]**2, 
                                                                               axis=1) \
                                                                 - tf.reduce_sum(tf.matmul(self._batched_F_op[b], 
                                                                                           tf.gather(self._Q_H, 
                                                                                                     self._H_Kinds, 
                                                                                                     axis=1))**2, 
                                                                                 axis=1))) / self.N_train 
                                      for b in range(self.batch_size_F)]
        return self._batched_var_H_op
  

    def get_batched_var_H_err_op(self):
        """
        Implements an op for the approximation error of the predictive variance 
        using the approximate Hessian estimator. 
        [Equation (20) using equation (11) in the paper].
        
        Args:
            None
            
        Returns:
            tensor of shape (batch_size_F, TL)
        """
        if self._Q_H is None or self._Lambda_H is None:
            self.init_eig_H_op(*self.compute_eig_H())
        if self._batched_F_op is None:
            self._batched_F_op = self.get_batched_F_op()
        if self._batched_var_H_err_op is None:
            self._batched_var_H_err_op = [self._lambda_H_gap_err_inv * (tf.reduce_sum(self._batched_F_op[b]**2, 
                                                                                      axis=1) \
                                                                        - tf.reduce_sum(tf.matmul(self._batched_F_op[b], 
                                                                                                  tf.gather(self._Q_H, 
                                                                                                            self._H_Kinds, 
                                                                                                            axis=1))**2,
                                                                                        axis=1)) / (self.N_train*2) 
                                          for b in range(self.batch_size_F)]
        return self._batched_var_H_err_op
 

    def eval_batched_var_H_op(self, X, K=None, returnError=False, progress=True, 
                              full_rank=True):
        """
        Evaluate op for the predictive variance (and optionally variance error) 
        using the approximate Hessian estimator.
        [Equation (19/20) using equation (11) in the paper].
        
        Args:
            X: ndarray of input data
            K (optional): int, number (<=K) of eigenpairs to base the 
                          approximation on.
            returnError (optional): Return also error estimate(s) of the variance 
                                    estimate(s), defaults to False.
            progress (optional): Boolean, if True display progress information, 
                                 defaults to True.
            full_rank: Boolean, if True make estimator full-rank (default). 
                       Otherwise => low-rank & no approx. errors.
        
        Returns:
            ndarray of shape (2, X.shape[0], TL) or (X.shape[0], TL) depending 
            on the value of returnError. The error(s) will be stacked after the 
            variance if returnError is True.
        """
        N = np.shape(X)[0]
        B = int(N/self.batch_size_F)

        pLambda_H = self.eval_Lambda_H(K, which='plambda')
        nLambda_H = self.reg_lambda_val

        Kinds = self._init_Kinds(K)

        print(Kinds)

        if full_rank is True:
            _lambda_H_gap_inv = (nLambda_H**-1 + pLambda_H**-1)/2.0
        else:
            _lambda_H_gap_inv = 0.0

        print(pLambda_H)
        print(nLambda_H)
                            
        if self._batched_var_H_op is None:
            self._batched_var_H_op = self.get_batched_var_H_op()
                
        if returnError is True and full_rank is True:
            if self._batched_var_H_err_op is None:
                self._batched_var_H_err_op = self.get_batched_var_H_err_op()
            
            _lambda_H_gap_err_inv = (nLambda_H**-1 - pLambda_H**-1)/2.0
                                    
            return np.concatenate([self.tfsession.run([self._batched_var_H_op, 
                                                       self._batched_var_H_err_op], 
                                                      feed_dict={self.X:X[b*self.batch_size_F:(b+1)*self.batch_size_F],
                                                                 self._H_Kinds:Kinds, 
                                                                 self._lambda_H_gap_inv:_lambda_H_gap_inv, 
                                                                 self._lambda_H_gap_err_inv:_lambda_H_gap_err_inv,
                                                                 self.dropout_rate: self.dropout_rate_val})
                                   for b in self._progress(range(B), progress)], axis=1)
        else:
            return np.concatenate([self.tfsession.run(self._batched_var_H_op, 
                                                     feed_dict={self.X:X[b*self.batch_size_F:(b+1)*self.batch_size_F],
                                                                self._H_Kinds:Kinds,
                                                                self._lambda_H_gap_inv:_lambda_H_gap_inv,
                                                                self.dropout_rate: self.dropout_rate_val})
                                   for b in self._progress(range(B), progress)])


    def get_batched_var_G_op(self):
        """
        Implements an op for the predictive variance using the approximate OPG estimator.
        [Equation (19) using equation (12) in the paper].
        
        Args:
            None
            
        Returns:
            tensor of shape (batch_size_F, TL)            
        """
        if self._Q_G is None or self._Lambda_G is None:
            self.init_eig_G_op(*self.compute_eig_G())
        if self._batched_F_op is None:
            self._batched_F_op = self.get_batched_F_op()
        if self._batched_var_G_op is None:
            self._batched_var_G_op = [(tf.reduce_sum(tf.matmul(self._batched_F_op[b], 
                                                               tf.gather(self._Q_G, 
                                                                         self._G_Kinds, 
                                                                         axis=1))**2 \
                                                     / tf.gather(self._Lambda_G,
                                                                 self._G_Kinds,
                                                                 axis=0),
                                                     axis=1) \
                                      + self._lambda_G_gap_inv*(tf.reduce_sum(self._batched_F_op[b]**2, 
                                                                              axis=1) \
                                                                - tf.reduce_sum(tf.matmul(self._batched_F_op[b], 
                                                                                          tf.gather(self._Q_G, 
                                                                                          self._G_Kinds, 
                                                                                          axis=1))**2, 
                                                                                axis=1))) / self.N_train 
                                     for b in range(self.batch_size_F)]
        return self._batched_var_G_op

  
    def get_batched_var_G_err_op(self):
        """
        Implements an op for the approximation error of the predictive variance
        using the approximate OPG estimator. 
        [Equation (20) using equation (12) in the paper].
        
        Args:
            None
            
        Returns:
            tensor of shape (batch_size_F, TL)
        """
        if self._Q_G is None or self._Lambda_G is None:
            self.init_eig_G_op(*self.compute_eig_G())
        if self._batched_F_op is None:
            self._batched_F_op = self.get_batched_F_op()
        if self._batched_var_G_err_op is None:
            self._batched_var_G_err_op = [self._lambda_G_gap_err_inv * (tf.reduce_sum(self._batched_F_op[b]**2, 
                                                                                      axis=1) \
                                                                        - tf.reduce_sum(tf.matmul(self._batched_F_op[b], 
                                                                                                  tf.gather(self._Q_G, 
                                                                                                            self._G_Kinds, 
                                                                                                            axis=1))**2,
                                                                                        axis=1)) / (self.N_train*2) 
                                          for b in range(self.batch_size_F)]
        return self._batched_var_G_err_op


    def eval_batched_var_G_op(self, X, K=None, returnError=False, progress=True,
                              full_rank=True):
        """
        Evaluate op for the predictive variance (and optionally variance error) 
        using the approximate OPG estimator.  
        [Equation (19/20) using equation (12) in the paper].
        
        Args:
            X: ndarray of input data
            K (optional): int, number (<=K) of eigenpairs to base the approximation 
                          on.
            returnError (optional): Return also error estimate(s) of the variance 
                                    estimate(s), defaults to False.
            progress (optional): Boolean, if True display progress information, 
                                 defaults to True.
            full_rank: Boolean, if True make estimator full-rank (default). 
                       Otherwise => low-rank & no approx. errors.
        
        Returns:
            ndarray of shape (2, X.shape[0], TL) or (X.shape[0], TL) depending on 
            the value of returnError. The error(s) will be stacked after the 
            variance if returnError is True.
        """

        N = np.shape(X)[0]
        B = int(N/self.batch_size_F)

        pLambda_G = self.eval_Lambda_G(K, which='plambda')
        nLambda_G = self.reg_lambda_val

        Kinds = self._init_Kinds(K)

        if full_rank is True:
            _lambda_G_gap_inv = (nLambda_G**-1 + pLambda_G**-1)/2.0
        else:
            _lambda_G_gap_H_gap_inv = 0.0
                            
        if self._batched_var_G_op is None:
            self._batched_var_G_op = self.get_batched_var_G_op()
                
        if returnError is True and full_rank is True:
            if self._batched_var_G_err_op is None:
                self._batched_var_G_err_op = self.get_batched_var_G_err_op()
            
            _lambda_G_gap_err_inv = (nLambda_G**-1 - pLambda_G**-1)/2.0

            return np.concatenate([self.tfsession.run([self._batched_var_G_op, 
                                                       self._batched_var_G_err_op], 
                                                      feed_dict={self.X:X[b*self.batch_size_F:(b+1)*self.batch_size_F],
                                                                 self._G_Kinds:Kinds, 
                                                                 self._lambda_G_gap_inv:_lambda_G_gap_inv, 
                                                                 self._lambda_G_gap_err_inv:_lambda_G_gap_err_inv,
                                                                 self.dropout_rate: self.dropout_rate_val})
                                   for b in self._progress(range(B), progress)], axis=1)
        else:
            return np.concatenate([self.tfsession.run(self._batched_var_G_op, 
                                                      feed_dict={self.X:X[b*self.batch_size_F:(b+1)*self.batch_size_F],
                                                                 self._G_Kinds:Kinds,
                                                                 self._lambda_G_gap_inv:_lambda_G_gap_inv,
                                                                 self.dropout_rate: self.dropout_rate_val})
                                   for b in self._progress(range(B), progress)])


    def get_batched_var_Sandwich_op(self):
        """
        Implements an op for the predictive variance using the approximate Sandwich estimator.
        [Equation (19) using equation (13) in the paper]. Note: The current implementation
        runs in float64 to avoid rounding errors leading to undefined behaviour.
        
        Args:
            None
            
        Returns:
            tensor of shape (batch_size_F, TL)            
        """
                
        if self._batched_F_op is None:
            self._batched_F_op = self.get_batched_F_op()

        F = self._batched_F_op
        Q_H = tf.gather(self._Q_H, self._H_Kinds, axis=1)
        Q_G = tf.gather(self._Q_G, self._G_Kinds, axis=1)
        inv_L_H = tf.cast(tf.diag(tf.gather(1.0/self._Lambda_H, self._H_Kinds, axis=0)), dtype='float64')
        L_G = tf.cast(tf.diag(tf.gather(self._Lambda_G, self._G_Kinds, axis=0)), dtype='float64')
        FQ_H = tf.cast(F@Q_H, dtype='float64')
        FQ_G = tf.cast(F@Q_G, dtype='float64')

        Q_H_T_F_T = [tf.transpose(FQ_H[b]) for b in range(self.batch_size_F)]
        Q_G_T_F_T = [tf.transpose(FQ_G[b]) for b in range(self.batch_size_F)]
     
        Q_H_T_Q_G = tf.cast(tf.matmul(Q_H, Q_G, transpose_a=True), dtype='float64')
        Q_G_T_Q_H = tf.transpose(Q_H_T_Q_G)
        Q_H_T_Q_H = tf.cast(tf.matmul(Q_H, Q_H, transpose_a=True), dtype='float64')
        
        Q_H_T_Q_G_Q_G_T_Q_H = Q_H_T_Q_G@Q_G_T_Q_H
        L_G_Q_G_T_Q_H = L_G@Q_G_T_Q_H

        _s = [tf.diag_part(FQ_H[b]@inv_L_H@Q_H_T_Q_G@L_G_Q_G_T_Q_H@inv_L_H@Q_H_T_F_T[b]) 
              for b in range(self.batch_size_F)]
        _a = [tf.diag_part(FQ_H[b]@inv_L_H@Q_H_T_Q_H@inv_L_H@Q_H_T_F_T[b] \
              - FQ_H[b]@inv_L_H@Q_H_T_Q_G_Q_G_T_Q_H@inv_L_H@Q_H_T_F_T[b]) 
              for b in range(self.batch_size_F)]
        _n = [tf.diag_part(FQ_G[b]@L_G_Q_G_T_Q_H@inv_L_H@Q_H_T_F_T[b] \
              - FQ_H[b]@Q_H_T_Q_G@L_G_Q_G_T_Q_H@inv_L_H@Q_H_T_F_T[b]) 
              for b in range(self.batch_size_F)]
        _d = [tf.diag_part(FQ_H[b]@inv_L_H@Q_H_T_F_T[b] \
              - FQ_G[b]@Q_G_T_Q_H@inv_L_H@Q_H_T_F_T[b] \
              - FQ_H[b]@Q_H_T_Q_H@inv_L_H@Q_H_T_F_T[b] \
              + FQ_H[b]@Q_H_T_Q_G_Q_G_T_Q_H@inv_L_H@Q_H_T_F_T[b]) 
              for b in range(self.batch_size_F)]
        _w = _n
        _i = _d
        _c = [tf.diag_part(FQ_G[b]@L_G@Q_G_T_F_T[b] \
              - FQ_G[b]@L_G_Q_G_T_Q_H@Q_H_T_F_T[b] \
              - FQ_H[b]@Q_H_T_Q_G@L_G@Q_G_T_F_T[b] \
              + FQ_H[b]@Q_H_T_Q_G@L_G_Q_G_T_Q_H@Q_H_T_F_T[b]) 
              for b in range(self.batch_size_F)]
        _h = [tf.diag_part(tf.cast(F[b]@tf.transpose(F[b]), dtype='float64') \
              - 2*FQ_H[b]@Q_H_T_F_T[b] - FQ_G[b]@Q_G_T_F_T[b] \
              + FQ_G[b]@Q_G_T_Q_H@Q_H_T_F_T[b] + FQ_H[b]@Q_H_T_Q_H@Q_H_T_F_T[b] \
              + FQ_H[b]@Q_H_T_Q_G@Q_G_T_F_T[b] - FQ_H[b]@Q_H_T_Q_G_Q_G_T_Q_H@Q_H_T_F_T[b]) 
              for b in range(self.batch_size_F)]

        if self._batched_var_Sandwich_op is None:
            self._batched_var_Sandwich_op = [(_s[b] + tf.cast(self._lambda_G_gap, dtype='float64')*_a[b] \
                                             + tf.cast(self._lambda_H_gap_inv, dtype='float64')*_n[b] \
                                             + tf.cast(self._lambda_H_gap_inv*self._lambda_G_gap, dtype='float64')*_d[b] \
                                             + tf.cast(self._lambda_H_gap_inv, dtype='float64')*_w[b] \
                                             + tf.cast(self._lambda_G_gap*self._lambda_H_gap_inv, dtype='float64')*_i[b] \
                                             + tf.cast(self._lambda_H_gap_inv*self._lambda_H_gap_inv, dtype='float64')*_c[b] \
                                             + tf.cast(self._lambda_H_gap_inv*self._lambda_G_gap*self._lambda_H_gap_inv, dtype='float64')*_h[b]) / self.N_train 
                                             for b in range(self.batch_size_F)]

        if self._batched_var_Sandwich_err_op is None:
            self._batched_var_Sandwich_err_op = [(tf.cast(self._plambda_G - self.reg_lambda_val, dtype='float64')*_a[b] \
                                                 + tf.cast(self.reg_lambda_val**-1 - self._plambda_H**-1, dtype='float64')*_n[b] \
                                                 + tf.cast(self.reg_lambda_val**-1*self._plambda_G - self._plambda_H**-1*self.reg_lambda_val, dtype='float64')*_d[b] \
                                                 + tf.cast(self.reg_lambda_val**-1 - self._plambda_H**-1, dtype='float64')*_w[b] \
                                                 + tf.cast(self.reg_lambda_val**-1*self._plambda_G - self._plambda_H**-1*self.reg_lambda_val, dtype='float64')*_i[b] \
                                                 + tf.cast(self.reg_lambda_val**-2 - self._plambda_H**-2, dtype='float64')*_c[b] \
                                                 + tf.cast(self.reg_lambda_val**-2*self._plambda_G - self._plambda_H**-2*self.reg_lambda_val, dtype='float64')*_h[b]) / (2*self.N_train) 
                                                 for b in range(self.batch_size_F)]
                                                 
        return self._batched_var_Sandwich_op, self._batched_var_Sandwich_err_op


    def eval_batched_var_Sandwich_op(self, X, K=None, returnError=False, 
                                     progress=True, full_rank=True):
        """
        Evaluate op for the predictive variance (and optionally variance error) 
        using the approximate Sandwich estimator.  
        [Equation (19/20) using equation (13) in the paper].
        
        Args:
            X: ndarray of input data
            K (optional): int, number (<=K) of eigenpairs to base the approximation 
                          on.
            returnError (optional): Return also error estimate(s) of the variance 
                                    estimate(s), defaults to False.
            progress (optional): Boolean, if True display progress information, 
                                 defaults to True.
            full_rank: Boolean, if True make estimator full-rank (default). 
                       Otherwise => low-rank & no approx. errors.
        
        Returns:
            ndarray of shape (2, X.shape[0], TL) or (X.shape[0], TL) depending on 
            the value of returnError. The error(s) will be stacked after the 
            variance if returnError is True.
        """

        N = X.shape[0]
        B = int(N / self.batch_size_F)

        H_Kinds = self._init_Kinds(K)

        pLambda_H = self.eval_Lambda_H(K, which='plambda')
        nLambda_H = self.reg_lambda_val

        if full_rank is True:
            _lambda_H_gap_inv = (nLambda_H**-1 + pLambda_H**-1)/2.0
        else:
            _lambda_H_gap_inv = 0.0

        G_Kinds = self._init_Kinds(K)

        pLambda_G = self.eval_Lambda_G(K, which='plambda')
        nLambda_G = self.reg_lambda_val

        _lambda_G_gap = ((nLambda_G**-1 + pLambda_G**-1)/2.0)**-1

        if full_rank is False:
            _lambda_G_gap = 0.0                         

        if self._batched_var_Sandwich_op is None:
            self._batched_var_Sandwich_op, self._batched_var_Sandwich_err_op = self.get_batched_var_Sandwich_op()


        if returnError is True and full_rank is True:
            return np.concatenate([self.tfsession.run([self._batched_var_Sandwich_op, self._batched_var_Sandwich_err_op],
                                                       feed_dict={self.X:X[b*self.batch_size_F:(b+1)*self.batch_size_F],  
                                                                  self.reg_lambda:self.reg_lambda_val,
                                                                  self._H_Kinds:H_Kinds,
                                                                  self._G_Kinds:G_Kinds,
                                                                  self._lambda_G_gap:_lambda_G_gap,
                                                                  self._lambda_H_gap_inv: _lambda_H_gap_inv,
                                                                  self._plambda_H: pLambda_H,
                                                                  self._plambda_G: pLambda_G})
                                   for b in self._progress(range(B), progress)], axis=1)
        else:
            return np.concatenate([self.tfsession.run(self._batched_var_Sandwich_op, 
                                                      feed_dict={self.X:X[b*self.batch_size_F:(b+1)*self.batch_size_F],  
                                                                 self.reg_lambda:self.reg_lambda_val,
                                                                 self._H_Kinds:H_Kinds,
                                                                 self._G_Kinds:G_Kinds,
                                                                 self._lambda_G_gap:_lambda_G_gap,
                                                                 self._lambda_H_gap_inv: _lambda_H_gap_inv})
                                   for b in self._progress(range(B), progress)])
