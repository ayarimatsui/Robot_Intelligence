#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from functions import *

class ReLU:
    def __init__(self):
        self.filter=None

    def forward(self,Z):
        self.filter=(Z<=0)
        X=Z.copy()
        X[self.filter]=0
        return X

    def backward(self,error):
        error[self.filter]=0
        next_error=error
        return next_error


class Sigmoid:
    def __init__(self):
        self.Y=None #出力

    def forward(self,x):
        self.Y=sigmoid(x)
        return self.Y

    def backward(self,error):
        next_error=error*self.Y*(1.0-self.Y)
        return next_error


class Affine:
    def __init__(self, weight, bias):
        self.W=weight
        self.b=bias
        self.X=None
        self.dX=None #Xの微分
        self.dW=None #Wの微分
        self.db=None #bの微分

    def forward(self,x):
        self.X=x
        z=np.dot(self.x,self.W)+self.b
        return z

    def backward(self,error):
        self.dX=np.dot(error,self.W.T)
        self.dW=np.dot(self.X.T,error)
        self.dB=np.sum(error,axis=0)
        next_error = self.dX
        return next_error


class SoftmaxWithCrossEntropyError:
    def __init__(self):
        self.loss=None #誤差(損失)
        self.Y=None #出力
        self.T=None #教師データラベル

    def forward(self,x,t):
        self.T=t
        self.Y=softmax(x)
        self.loss=cross_entropy_error(self.Y,self.T)
        return self.loss

    def backward(self):
        batch_size=self.T.shape[0]
        dZ=(self.Y-self.T)/batch_size
        return dZ
