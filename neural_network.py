#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from layers import *
from collections import OrderedDict

class NeuralNetWork:

    def __init__(self,input_size=784,hidden_size_list=[100],output_size=10,weight_init_std=0.1,load_params_dict=None):
        self.parameters={}
        self.parameters['W1']=weight_init_std*np.random.randn(input_size,hidden_size_list[0])
        self.parameters['b1']=np.zeros(hidden_size_list[0])
        self.parameters['W2']=weight_init_std*np.ramdom.randn(hidden_size_list[0],hidden_size_list[0])
        self.parameters['b2']=np.zeros(hidden_size_list[0])
        self.parameters['W3']=weight_init_std*np.random.randn(hidden_size_list[0],output_size)
        self.parameters['b3']=np.zeros(output_size)

        self.layers=OrderedDict()
        self.layers['Affine1']=Affine(self.parameters['W1'],self.parameters['b1'])
        self.layers['ReLU']=ReLU()  #変えてみる
        self.layers['Affine2']=Affine(self.parameters['W2'],self.parameters['b2'])
        self.layers['Sigmoid']=Sigmoid()  #変えてみる
        self.layers['Affine3']=Affine(self.parameters['W3'],self.parameters['b3'])
        self.output_layer=SoftmaxWithCrossEntropyError()


    def predict(self,X):
        for layer in self.layers.values():
            X=layer.forward(X)
        Z=X
        return Z


    def predicted_label(self,X):
        Z=self.predict(X)
        Y=np.argmax(Z,axis=1)
        return Y


    def loss(self,X,T):
        Z=self.predict(X)
        return  self.output_layer.forward(Z,T)


    def accuracy(self,X,T):
        Z=self.predict(X)
        Y=np.argmax(Z,axis=1)
        data_num=X.shape[0]
        if T.ndim!=1:
            T=np.argmax(T,axis=1)
        accuracy=np.sum(Y==T)/float(data_num)

        return accuracy


    def grad(self,X,T):
        #FP
        self.loss(X,T)
        #BP
        error=self.output_layer.backward()
        layers_reverse=list(self.layers.values())
        layers_reverse.reverse()
        for layer in layers_reverse:
            error=layer.backward(error)

        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db
        grads['W3']=self.layers['Affine3'].dW
        grads['b3']=self.layers['Affine3'].db

        return grads
