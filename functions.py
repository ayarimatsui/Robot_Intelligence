#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

# 活性化関数とその微分関数
def ReLU(x):
    return np.maximum(0,x)

def grad_ReLU(x):
    grad=np.zeros_like(x)
    grad[x>=0]=1.0
    return grad

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def grad_sigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def softmax(x):
    if x.ndim==2:
        x=x.T
        x=x-np.max(x,axis=0)
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    else:
        y=np.exp(x-np.max(x))
        return y/np.sum(y)


# 誤差評価関数
#今回のニューラルネットワークモデルでは、多クラス分類を行いたいので、交差エントロピー誤差関数を用いる
def cross_entropy_error(y,t):
    # 1次元ベクトルの場合
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    batch_size=y.shape[0]
    delta=1e-7
    return -np.sum(t*np.log(y+delta))/batch_size
