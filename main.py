#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import *
from neural_network import *
from optimizer import *

class Training:
    def __init__(self, network, optimizer, train_imgs, train_labels, test_imgs, test_labels, dataset_loops, batch_size):
        self.network=network
        self.optimizer=optimizer
        self.train_imgs=train_imgs
        self.train_labels=train_labels
        self.test_imgs=test_imgs
        self.test_labels=test_labels
        self.train_accuracy_list=[]
        self.test_accuracy_list=[]
        self.train_loss_list=[]
        self.test_loss_list=[]
        self.batch_size=batch_size
        self.train_size=train_imgs.shape[0]
        self.dataset_loops=dataset_loops # データセットを何周するか
        self.iterators_per_loop=max(int(self.train_size/batch_size),1)
        self.total_loops=int(dataset_loops*self.iterators_per_loop) # バッチ学習を計何回するか
        self.current_iterator=0
        self.current_loop_index=0

    #学習（1ループ）
    def train(self):
        batch_index=np.random.choice(self.train_size,self.batch_size)
        img_batch=self.train_imgs[batch_index]
        label_batch=self.train_labels[batch_index]
        #勾配を求める
        grads=self.network.grad(img_batch,label_batch)
        #重みデータを更新
        self.network.parameters=self.optimizer.update(self.network.parameters,grads)
        self.network.updateLayers()

        #精度を計算しリストに格納する
        if self.current_iterator==0:
            train_accuracy=self.network.accuracy(self.train_imgs,self.train_labels)
            test_accuracy=self.network.accuracy(self.test_imgs,self.test_labels)
            self.train_accuracy_list.append(train_accuracy)
            self.test_accuracy_list.append(test_accuracy)
            train_loss=self.network.loss(self.train_imgs,self.train_labels)
            test_loss=self.network.loss(self.test_imgs,self.test_labels)
            self.train_loss_list.append(train_loss)
            self.test_loss_list.append(test_loss)
            print('train_accuracy: {}   test_accuracy: {}   train_loss: {}   test_loss: {}'.format(train_accuracy,test_accuracy,train_loss,test_loss))


    #学習開始関数
    def startTraining(self):
        for i in range(self.total_loops):
            self.train()
            self.current_iterator+=1
            if self.current_iterator>=self.iterators_per_loop:
                self.current_iterator=0
                self.current_loop_index+=1
                print('loop: {}/{} done'.format(self.current_loop_index,self.dataset_loops))
        print('training done')

    #グラフ化
    def showAccuracy(self):
        plt.plot(np.arange(self.dataset_loops+1)[1:],np.array(self.train_accuracy_list),label='Train Data')
        plt.plot(np.arange(self.dataset_loops+1)[1:],np.array(self.test_accuracy_list),label='Test Data')
        plt.legend(loc='lower right')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.ylim([0,100])
        plt.show()
        plt.clf()

    def showLoss(self):
        plt.plot(np.arange(self.dataset_loops+1)[1:],np.array(self.train_loss_list),label='Train Data')
        plt.plot(np.arange(self.dataset_loops+1)[1:],np.array(self.test_loss_list),label='Test Data')
        plt.legend(loc='lower right')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim([0,100])
        plt.show()
        plt.clf()


if __name__=="__main__":

    DataPreprocessing=DataPreprocessing()

    if not os.path.isfile("dataset/mnist.pkl"):
        DataPreprocessing.save_dataset()

    train_imgs,train_labels,test_imgs,test_labels=DataPreprocessing.preprocess_data()
    #train_imgs,train_labels,test_imgs,test_labels=DataPreprocessing.noise_added(25)

    network=NeuralNetWork()
    opt=SGD(learning_rate=0.01)

    training=Training(network=network, optimizer=opt, train_imgs=train_imgs, train_labels=train_labels, test_imgs=test_imgs, test_labels=test_labels, dataset_loops=100, batch_size=100) #dataset_loops変更可 50くらいでも良さそう

    training.startTraining()

    training.showAccuracy()
    training.showLoss()
