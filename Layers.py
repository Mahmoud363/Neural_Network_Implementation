from typing import Tuple, Optional
import skimage.io
from skimage.transform import resize
import os
import sys
from skimage.color import rgb2gray
import numpy as np
import cupy as cp
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle as pickle
from sklearn.utils import shuffle
from PIL import Image
import os.path
from PIL import ImageOps
from skimage.io import imread, imshow
import copy

from utils import random_seeded_4d, random_seeded_2d

# a base class for the rest of the layers to inherit from
class layer:
  def __init__(self):
    pass
  def forward(self, X):
    return input
  def backward(self, X, dout, training = True):
    return dout
  def regularize_forward(self):
    return 0
  def save(self):
    pass
  def load(self):
    pass
  def save_best(self):
    pass
  def load_best(self):
    pass

class MaxPoolLayer(layer):

    def __init__(self, pool_size: Tuple[int, int], stride: int = 2):
        """
        :param pool_size - tuple holding shape of 2D pooling window
        :param stride - stride along width and height of input volume used to
        apply pooling operation
        """
        self._pool_size = pool_size
        self._stride = stride
        self._a = None
        self._cache = {}

    def forward(self, a_prev: cp.array, training = True) -> cp.array:
        """
        :param a_prev - 4D tensor with shape(n, h_in, w_in, c)
        :output 4D tensor with shape(n, h_out, w_out, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        c - number of channels of the input/output volume
        w_out - width of output volume
        h_out - width of output volume
        """
        self._a = cp.array(a_prev, copy=True)
        n, h_in, w_in, c = a_prev.shape
        h_pool, w_pool = self._pool_size
        h_out = 1 + (h_in - h_pool) // self._stride
        w_out = 1 + (w_in - w_pool) // self._stride
        output = cp.zeros((n, h_out, w_out, c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                a_prev_slice = a_prev[:, h_start:h_end, w_start:w_end, :]
                self._save_mask(x=a_prev_slice, cords=(i, j))
                output[:, i, j, :] = cp.max(a_prev_slice, axis=(1, 2))
        return output

    def backward(self, x: cp.array, da_curr: cp.array) -> cp.array:
        """
        :param da_curr - 4D tensor with shape(n, h_out, w_out, c)
        :output 4D tensor with shape(n, h_in, w_in, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        c - number of channels of the input/output volume
        w_out - width of output volume
        h_out - width of output volume
        """
        output = cp.zeros_like(self._a)
        _, h_out, w_out, _ = da_curr.shape
        h_pool, w_pool = self._pool_size

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_pool
                w_start = j * self._stride
                w_end = w_start + w_pool
                output[:, h_start:h_end, w_start:w_end, :] += \
                    da_curr[:, i:i + 1, j:j + 1, :] * self._cache[(i, j)]
        return output

    def _save_mask(self, x: cp.array, cords: Tuple[int, int]) -> None:
        mask = cp.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = cp.argmax(x, axis=1)

        n_idx, c_idx = cp.indices((n, c))
        mask.reshape((n, h * w, c))[n_idx, idx, c_idx] = 1
        self._cache[cords] = mask

class FlattenLayer(layer):

    def __init__(self):
        self._shape = ()

    def forward(self, a_prev: cp.array, training = True) -> cp.array:
        """
        :param a_prev - ND tensor with shape (n, ..., channels)
        :output - 1D tensor with shape (n, 1)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        self._shape = a_prev.shape
        return cp.ravel(a_prev).reshape(a_prev.shape[0], -1)

    def backward(self, x: cp.array, da_curr: cp.array) -> cp.array:
        """
        :param da_curr - 1D tensor with shape (n, 1)
        :output - ND tensor with shape (n, ..., channels)
        ------------------------------------------------------------------------
        n - number of examples in batch
        """
        return da_curr.reshape(self._shape)

class ConvLayer2D(layer):

    def __init__(self, filter_size=(3,3,3),filters_num=16, padding = 'same',stride = 1,
            lr = 0.001, reg = 0, opt='nadam', prefix='best_', name='Conv2d'):
        """
        :param w -  4D tensor with shape (h_f, w_f, c_f, n_f)
        :param b - 1D tensor with shape (n_f, )
        :param padding - flag describing type of activation padding valid/same
        :param stride - stride along width and height of input volume
        ------------------------------------------------------------------------
        h_f - height of filter volume
        w_f - width of filter volume
        c_f - number of channels of filter volume
        n_f - number of filters in filter volume
        """
        self.lr=lr 
        self.reg = reg
        self.V=0
        self.t=1
        self.Acc=0
        self.mo=0
        self.opt = opt
        self.prefix=prefix
        self.name = name
        h_f, w_f, c_f = filter_size
        self._w = random_seeded_4d(h_f, w_f, c_f, filters_num) /cp.sqrt(2*h_f* w_f*c_f*filters_num)
        self._b =  cp.zeros(filters_num) 
        self._padding = padding
        self._stride = stride
        self._dw, self._db = None, None
        self.a_prev = None

    def forward(self, a_prev: cp.array, training = True):
        """
        :param a_prev - 4D tensor with shape (n, h_in, w_in, c)
        :output 4D tensor with shape (n, h_out, w_out, n_f)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        #self.a_prev = cp.array(a_prev, copy=True)
        output_shape = self.calculate_output_dims(input_dims=a_prev.shape)
        n, h_in, w_in, _ = a_prev.shape
        _, h_out, w_out, _ = output_shape
        h_f, w_f, _, n_f = self._w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=a_prev, pad=pad)
        output = cp.zeros(output_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f

                output[:, i, j, :] = cp.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, cp.newaxis] *
                    self._w[cp.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )

        return output + self._b

    def backward(self, a_prev: cp.array, da_curr: cp.array):
        """
        :param da_curr - 4D tensor with shape (n, h_out, w_out, n_f)
        :output 4D tensor with shape (n, h_in, w_in, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        _, h_out, w_out, _ = da_curr.shape
        n, h_in, w_in, _ = a_prev.shape
        h_f, w_f, _, _ = self._w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=a_prev, pad=pad)
        output = cp.zeros_like(a_prev_pad)

        self._db = da_curr.sum(axis=(0, 1, 2)) / n
        self._dw = cp.zeros_like(self._w)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f
                output[:, h_start:h_end, w_start:w_end, :] += cp.sum(
                    self._w[cp.newaxis, :, :, :, :] *
                    da_curr[:, i:i+1, j:j+1, cp.newaxis, :],
                    axis=4
                )
                self._dw += cp.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, cp.newaxis] *
                    da_curr[:, i:i+1, j:j+1, cp.newaxis, :],
                    axis=0
                )
        self._dw += self.reg*self._w
        self._dw /= n
        self.update(self._dw, self._db)
        return output[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in, :]
    
    def regularize_forward(self): # compute the layer regularization
        return 0.5 * self.reg * cp.sum(cp.square(self._w))

    def update(self, dw, db):
        out = self.optimize(dw)
        self._w+=out
        self._b= self._b - self.lr*db 

    def optimize(self, dw):
        if self.opt=='momentum':
            return 0.6*self.V-self.lr*dw
        elif self.opt=='adagrad':
            self.Acc+=dw*dw
            return -self.lr*dw/(cp.sqrt(self.Acc)+1e-7)
        elif self.opt=='rmsprop':
            decay=0.2
            self.Acc=decay*self.Acc+(1-decay)*dw*dw
            return -self.lr*dw/(cp.sqrt(self.Acc)+1e-7)
        elif self.opt=='adam':
            beta1=0.6
            beta2=0.2
            self.mo=beta1*self.mo+(1-beta1)*dw
            self.Acc=beta2*self.Acc + (1-beta2)*dw*dw
            return -self.lr*self.mo/(cp.sqrt(self.Acc)+1e-7)
        elif self.opt == 'nadam':
            self.t+=1
            beta1=0.6
            beta2=0.2
            self.mo=beta1*self.mo+(1-beta1)*dw
            self.mo= self.mo / (1 - cp.power(beta1, self.t))
            self.Acc=beta2*self.Acc + (1-beta2)*dw*dw
            self.Acc = self.Acc / (1 - cp.power(beta1, self.t))
            return -self.lr*self.mo/(cp.sqrt(self.Acc)+1e-7)

    def save_best(self):
        self.best_w = copy.deepcopy(self._w)
        self.best_b = copy.deepcopy(self._b)

    def load_best(self):
        self._w = copy.deepcopy(self.best_w)
        self._b = copy.deepcopy(self.best_b)

    def save(self):
        path_w = self.prefix+self.name+'_w.npy'
        path_b = self.prefix+self.name+'_b.npy'
        with open(path_w, 'wb') as f:
            cp.save(f, self._w)
        with open(path_b, 'wb') as f:
            cp.save(f, self._b)

    def load(self):
        path_w = self.prefix+self.name+'_w.npy'
        path_b = self.prefix+self.name+'_b.npy'
        with open(path_w, 'rb') as f:
            self._w = cp.load(f)
        with open(path_b, 'rb') as f:
            self._b = cp.load(f)
        
    def calculate_output_dims(
        self, input_dims: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        :param input_dims - 4 element tuple (n, h_in, w_in, c)
        :output 4 element tuple (n, h_out, w_out, n_f)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        n, h_in, w_in, _ = input_dims
        h_f, w_f, _, n_f = self._w.shape
        if self._padding == 'same':
            return n, h_in, w_in, n_f
        elif self._padding == 'valid':
            h_out = (h_in - h_f) // self._stride + 1
            w_out = (w_in - w_f) // self._stride + 1
            return n, h_out, w_out, n_f
       

    def calculate_pad_dims(self) -> Tuple[int, int]:
        """
        :output - 2 element tuple (h_pad, w_pad)
        ------------------------------------------------------------------------
        h_pad - single side padding on height of the volume
        w_pad - single side padding on width of the volume
        """
        if self._padding == 'same':
            h_f, w_f, _, _ = self._w.shape
            return (h_f - 1) // 2, (w_f - 1) // 2
        elif self._padding == 'valid':
            return 0, 0
        

    @staticmethod
    def pad(array: cp.array, pad: Tuple[int, int]) -> cp.array:
        """
        :param array -  4D tensor with shape (n, h_in, w_in, c)
        :param pad - 2 element tuple (h_pad, w_pad)
        :output 4D tensor with shape (n, h_out, w_out, n_f)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        h_pad - single side padding on height of the volume
        w_pad - single side padding on width of the volume
        """
        return cp.pad(
            array=array,
            pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
            mode='constant'
        )




# a fully connected dense layer
# can be created with different paramaters and optimizers
class Dense(layer):
    def __init__(self, input_size, out_size, lr=0.001,
        reg=0.0, use_batchnorm = False, opt='nadam', prefix='best_', name='Dense'): 
        self.lr=lr 
        self.reg = reg
        self.V=0
        self.t=1
        self.Acc=0
        self.mo=0
        self.opt = opt
        self.use_batchnorm = use_batchnorm
        self.prefix=prefix
        self.name = name
        cp.random.seed(42)
        self.weights=cp.asarray(random_seeded_2d(input_size, out_size)/np.sqrt(input_size))
        self.b= cp.zeros(out_size)
        self.best = None

    def forward(self, X, training = True): #X.shape= (n. of images, n. of features) = (btachsize, 3072)
        out = cp.dot(X, self.weights)+ self.b
        return out

    def backward(self, X, dout):
        gIn=cp.dot(dout, self.weights.T) # input gradient
        dw= cp.dot(X.T, dout) + self.reg*self.weights # weights gradient
        db= dout.mean(axis=0)*X.shape[0] # bias gradient
        self.update(dw, db) # call the optimizer
        return gIn

    def regularize_forward(self): # compute the layer regularization
        return 0.5 * self.reg * cp.sum(cp.square(self.weights))

    def update(self, dw, db):
        out = self.optimize(dw)
        self.weights+=out
        self.b= self.b - self.lr*db 

    def optimize(self, dw):
        if self.opt=='momentum':
            return 0.6*self.V-self.lr*dw
        elif self.opt=='adagrad':
            self.Acc+=dw*dw
            return -self.lr*dw/(cp.sqrt(self.Acc)+1e-7)
        elif self.opt=='rmsprop':
            decay=0.2
            self.Acc=decay*self.Acc+(1-decay)*dw*dw
            return -self.lr*dw/(cp.sqrt(self.Acc)+1e-7)
        elif self.opt=='adam':
            beta1=0.6
            beta2=0.2
            self.mo=beta1*self.mo+(1-beta1)*dw
            self.Acc=beta2*self.Acc + (1-beta2)*dw*dw
            return -self.lr*self.mo/(cp.sqrt(self.Acc)+1e-7)
        elif self.opt == 'nadam':
            self.t+=1
            beta1=0.6
            beta2=0.2
            self.mo=beta1*self.mo+(1-beta1)*dw
            self.mo= self.mo / (1 - cp.power(beta1, self.t))
            self.Acc=beta2*self.Acc + (1-beta2)*dw*dw
            self.Acc = self.Acc / (1 - cp.power(beta1, self.t))
            return -self.lr*self.mo/(cp.sqrt(self.Acc)+1e-7)

    def save_best(self):
        self.best_w = copy.deepcopy(self.weights)
        self.best_b = copy.deepcopy(self.b)

    def load_best(self):
        self.weights = copy.deepcopy(self.best_w)
        self.b = copy.deepcopy(self.best_b)

    def save(self):
        path_w = self.prefix+self.name+'_w.npy'
        path_b = self.prefix+self.name+'_b.npy'
        with open(path_w, 'wb') as f:
            cp.save(f, self.weights)
        with open(path_b, 'wb') as f:
            cp.save(f, self.b)

    def load(self):
        path_w = self.prefix+self.name+'_w.npy'
        path_b = self.prefix+self.name+'_b.npy'
        with open(path_w, 'rb') as f:
            self.weights = cp.load(f)
        with open(path_b, 'rb') as f:
            self.b = cp.load(f)
    
    

class BatchNorm(layer):
    def __init__(self, out_size, lr=0.001, opt='nadam',
    prefix='best_', name='BatchNorm'): 
        self.lr=lr 
        self.opt = opt
        self.V=0
        self.t=1
        self.Acc=0
        self.mo=0
        self.running_mean = cp.zeros(out_size)
        self.running_var = cp.zeros(out_size)
        self.gamma = cp.ones(out_size)
        self.beta = cp.zeros(out_size)
        self.best_gamma = None
        self.best_beta = None
        self.prefix=prefix
        self.name = name

    def forward(self, x, training = True):
        eps =  1e-5
        momentum = 0.9
        N, D = x.shape
        running_mean = self.running_mean
        running_var = self.running_var
        out = None

        mu = cp.mean(x, axis=0)

        var = 1 / float(N) * cp.sum((x - mu) ** 2, axis=0)
        x_hat = (x - mu) / cp.sqrt(var + eps)
        y = self.gamma * x_hat + self.beta
        out = y
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        self.x_hat, self.mu, self.var, self.eps, self.x =x_hat, mu, var, eps, x

        self.running_mean = running_mean
        self.running_var = running_var
        return out

    def backward(self, X, dout):
        x_hat, mu, var, eps, gamma, beta, x = self.x_hat, self.mu, self.var, self.eps, self.gamma, self.beta, self.x
        
        N, D = dout.shape
        dbeta = cp.sum(dout, axis=0)
        dgamma = cp.sum(dout * x_hat, axis=0)
        dx_hat = dout * gamma
        dxmu1 = dx_hat * 1 / cp.sqrt(var + eps)
        divar = cp.sum(dx_hat * (x - mu), axis=0)
        dvar = divar * -1 / 2 * cp.power((var + eps), (-3/2))
        dsq = 1 / N * cp.ones((N, D)) * dvar
        dxmu2 = 2 * (x - mu) * dsq
        dx1 = dxmu1 + dxmu2
        dmu = -1 * cp.sum(dxmu1 + dxmu2, axis=0)
        dx2 = 1 / N * cp.ones((N, D)) * dmu
        dx = dx1 + dx2

        self.update(dgamma, dbeta) 
        return dx

    def update(self, dgamma, dbeta):
        out1 = self.optimize(dgamma)
        self.gamma+=out1
        out2 = self.optimize(dbeta)
        self.beta+=out2

    def optimize(self, dw):
        if self.opt=='momentum':
            return 0.6*self.V-self.lr*dw
        elif self.opt=='adagrad':
            self.Acc+=dw*dw
            return -self.lr*dw/(cp.sqrt(self.Acc)+1e-7)
        elif self.opt=='rmsprop':
            decay=0.2
            self.Acc=decay*self.Acc+(1-decay)*dw*dw
            return -self.lr*dw/(cp.sqrt(self.Acc)+1e-7)
        elif self.opt=='adam':
            beta1=0.6
            beta2=0.2
            self.mo=beta1*self.mo+(1-beta1)*dw
            self.Acc=beta2*self.Acc + (1-beta2)*dw*dw
            return -self.lr*self.mo/(cp.sqrt(self.Acc)+1e-7)
        elif self.opt == 'nadam':
            self.t+=1
            beta1=0.6
            beta2=0.2
            self.mo=beta1*self.mo+(1-beta1)*dw
            self.mo= self.mo / (1 - cp.power(beta1, self.t))
            self.Acc=beta2*self.Acc + (1-beta2)*dw*dw
            self.Acc = self.Acc / (1 - cp.power(beta1, self.t))
            return -self.lr*self.mo/(cp.sqrt(self.Acc)+1e-7)

    def save_best(self):
        self.best_gamma = copy.deepcopy(self.gamma)
        self.best_beta = copy.deepcopy(self.beta)
    
    def load_best(self):
        self.gamma = copy.deepcopy(self.best_gamma)
        self.beta = copy.deepcopy(self.best_beta)

    def save(self):
        path_gamma = self.prefix+self.name+'_gamma.npy'
        path_beta = self.prefix+self.name+'_beta.npy'
        with open(path_gamma, 'wb') as f:
            cp.save(f, self.gamma)
        with open(path_beta, 'wb') as f:
            cp.save(f, self.beta)
    
    def load(self):
        path_gamma = self.prefix+self.name+'_gamma.npy'
        path_beta = self.prefix+self.name+'_beta.npy'
        with open(path_gamma, 'rb') as f:
            self.gamma = cp.load(f)
        with open(path_beta, 'rb') as f:
            self.beta = cp.load(f)


class ReLU(layer): 
  def __init__(self):
      pass

  def forward(self, X, training = True):
      relUF= cp.maximum(0, X)
      return relUF

  def backward(self, X, dout):
      dz= X>0
      return dout*dz

class Dropout(layer): 
  def __init__(self, p_dropout):
      self.p_dropout = p_dropout

  def forward(self, X, training = True):
    if training == True:
        u = cp.random.binomial(1, self.p_dropout, size=X.shape) / self.p_dropout
        out = X * u
        self.cache = u
    else:
        out = X
    return out

  def backward(self, X, dout):
    dX = dout * self.cache
    return dX

class leakyReLU(layer): 
  def __init__(self):
      pass

  def forward(self, X, training = True):
      relUF= cp.maximum(X, 0.1*X)
      return relUF

  def backward(self, X, dout):
      return ((X>0)*1+(X<0)*0.)*dout

class Loss(object):
  def __init__(self, X, Y):
      self.Y=Y
      self.X=X
  def forward(self):
      pass
  def backward(self):
      pass

class SoftMax(Loss):
  def __init__(self, X, Y):
    self.Y=Y
    self.X=X-cp.max(X, axis=1, keepdims=True)
  def forward(self):
    scores=self.X[cp.arange(len(self.X)), self.Y]
    Loss=-scores+ cp.log(cp.sum(cp.exp(self.X), axis=1))
    return Loss
  def backward(self):
    df1=cp.zeros_like(self.X)
    df1[cp.arange(len(self.X)), self.Y]=1
    df2=cp.exp(self.X)/cp.exp(self.X).sum(axis=-1, keepdims=True)
    return (-df1+df2)/self.X.shape[0]
