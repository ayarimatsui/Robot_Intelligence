#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from common.network import *
from common.trainer import *
from common.optimizer import *
from dataset.dataset import *

dataset_directory = os.path.dirname(os.path.abspath(__file__))

mnist = Dataset(dataset_dir=dataset_directory,save_as='mnist',image_size=784,train_size=60000,test_size=10000,
                train_image_file='dataset/train-images-idx3-ubyte.gz',
                train_label_file='dataset/train-labels-idx1-ubyte.gz',
                test_image_file='dataset/t10k-images-idx3-ubyte.gz',
                test_label_file='dataset/t10k-labels-idx1-ubyte.gz')

if not os.path.exists(mnist.save_as):
    mnist.saveDataset()

x_train, t_train, x_test, t_test = mnist.loadDataset()

net = NetWork(input_size=784,output_size=10,hidden_layer_size_list=[100,100],activator='sigmoid',enable_weight_decay=False,weight_decay_lambda=None)
opt = SGD(learning_rate=0.01)

trainer = Trainer(network=net, optimizer=opt, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test, dataset_loops=50, batch_size=100)

trainer.startTraining()

trainer.showAccuracy()
