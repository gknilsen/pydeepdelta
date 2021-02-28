"""    
    utils.py - helper functions
    
    Copyright (c) 2018-2019 by Geir K. Nilsen (geir.kjetil.nilsen@gmail.com)
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

import numpy as np
import os
import tensorflow.compat.v1 as tf
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.disable_v2_behavior()

def delete_all_except(x, y):
    return x[[z for z in range(x.shape[0]) if z in [y]]]

def to_one_hot(y, n=None):
    if n is None:
        n = 10
    return np.array([np.eye(1, n, y[i])[0] for i in range(len(y))],dtype='float32')

def conv2d(x, W, b):
    # Conv2D wrapper
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')


def confuse_max(yhat, y, sigma, start, stop):
    TPx=TPy=FPx=FPy=np.array([])
    N = stop - start
    
    for n in range(N):
        # True Positive
        if np.argmax(yhat[n],axis=0) == np.argmax(y[n],axis=0):
            TPx = np.append(TPx, np.max(yhat[n],axis=0))
            TPy = np.append(TPy, sigma[n][np.argmax(yhat[n],axis=0)])
        # False Positive
        else:
            FPx = np.append(FPx, np.max(yhat[n],axis=0))
            FPy = np.append(FPy, sigma[n][np.argmax(yhat[n],axis=0)])

    return TPx, TPy, FPx, FPy


def confuse_score(yhat, y, sigma_score, start, stop):
    TPx=TPy=FPx=FPy=np.array([])
    N = stop - start
    
    for n in range(N):
        # True Positive
        if np.argmax(yhat[n],axis=0) == np.argmax(y[n],axis=0):
            TPx = np.append(TPx, np.max(yhat[n],axis=0))
            TPy = np.append(TPy, sigma_score[n])
        # False Positive
        else:
            FPx = np.append(FPx, np.max(yhat[n],axis=0))
            FPy = np.append(FPy, sigma_score[n])

    return TPx, TPy, FPx, FPy            

def confusePN(mean, se, start, stop, y, elems=None):
    Px=Py=Nx=Ny=np.array([])
    N = stop - start
    m = y.shape[1]
    for ex in range(N):
        Nx = np.append(Nx, np.delete(mean[ex], np.argmax(mean[ex])))
        Ny = np.append(Ny, np.delete(se[ex], np.argmax(mean[ex])))
    
        Px = np.append(Px, delete_all_except(mean[ex], np.argmax(mean[ex])))
        Py = np.append(Py, delete_all_except(se[ex], np.argmax(mean[ex])))

    return Px, Py, Nx, Ny

def confuse(mean, se, start, stop, y, elems=None):
    # Generate confusion "matrix"
    TPx=TPy=TNx=TNy=FPx=FPy=FNx=FNy=np.array([])
    N = stop - start
    m = y.shape[1]
    if elems is None:
        for n in range(N):
            ind = n + start
            pos = np.argmax(mean[ind,:],axis=0)
            neg = [x for x in range(m) if x != pos]
        
            true = np.argmax(y[ind,:], axis=0)
            
            if np.sum(y[ind,:]) == 0:
                TNx = np.append(TNx, mean[ind,neg])
                TNy = np.append(TNy, se[ind,neg])
                
                FPx = np.append(FPx, mean[ind,pos])
                FPy = np.append(FPy, se[ind,pos])
    
                continue
        
            if pos == true:
                TPx = np.append(TPx, mean[ind,pos])
                TPy = np.append(TPy, se[ind,pos])
                TNx = np.append(TNx, mean[ind,neg])
                TNy = np.append(TNy, se[ind,neg])
            
            if pos != true:
                false = [x for x in range(m) if x != pos and x != true]
                TNx = np.append(TNx, mean[ind,false])
                TNy = np.append(TNy, se[ind,false])
                
                FPx = np.append(FPx, mean[ind,pos])
                FPy = np.append(FPy, se[ind,pos])
                
                FNx = np.append(FNx, mean[ind,true])
                FNy = np.append(FNy, se[ind,true])
    
    else:
        for n in elems:
            ind = n + start
            pos = np.argmax(mean[ind,:],axis=0)
            neg = [x for x in range(m) if x != pos]
        
            true = np.argmax(y[ind,:], axis=0)
            
            if np.sum(y[ind,:]) == 0:
                TNx = np.append(TNx, mean[ind,neg])
                TNy = np.append(TNy, se[ind,neg])
                
                FPx = np.append(FPx, mean[ind,pos])
                FPy = np.append(FPy, se[ind,pos])
    
                continue
        
            if pos == true:
                TPx = np.append(TPx, mean[ind,pos])
                TPy = np.append(TPy, se[ind,pos])
                TNx = np.append(TNx, mean[ind,neg])
                TNy = np.append(TNy, se[ind,neg])
            
            if pos != true:
                false = [x for x in range(m) if x != pos and x != true]
                TNx = np.append(TNx, mean[ind,false])
                TNy = np.append(TNy, se[ind,false])
                
                FPx = np.append(FPx, mean[ind,pos])
                FPy = np.append(FPy, se[ind,pos])
                
                FNx = np.append(FNx, mean[ind,true])
                FNy = np.append(FNy, se[ind,true])
    
    return (TPx,TPy, TNx, TNy, FPx, FPy, FNx, FNy)

   
