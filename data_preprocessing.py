#! /usr/bin/env python
# -*- coding: utf-8 -*-
import urllib.request
import gzip
import numpy as np
import pickle
from numpy.random import *
import random

class DataPreprocessing:

    def __init__(self):
        # データのダウンロード
        url='http://yann.lecun.com/exdb/mnist/'
        self.data_file={
            'train_img':'train-images-idx3-ubyte.gz',
            'train_label':'train-labels-idx1-ubyte.gz',
            'test_img':'t10k-images-idx3-ubyte.gz',
            'test_label':'t10k-labels-idx1-ubyte.gz'
        }

        self.dataset_dir='dataset'

        for value in self.data_file.values():
            file_path=self.dataset_dir+'/'+value
            urllib.request.urlretrieve(url+value,file_path)

        self.dataset={}
        self.save_file=None


    def load_img(self,file_name):
        file_path=self.dataset_dir+'/'+file_name
        with gzip.open(file_path,'rb') as f:
            data=np.frombuffer(f.read(),np.uint8,offset=16) #データが余分にあるのでoffsetでカット
        data=data.reshape(-1,784)
        return data


    def load_label(self,file_name):
        file_path=self.dataset_dir+'/'+file_name
        with gzip.open(file_path,'rb') as f:
            labels=np.frombuffer(f.read(),np.uint8,offset=8) #データが余分にあるのでoffsetでカット
        return labels


    def save_dataset(self): #必ずmainファイルで実行
        self.dataset['train_img']=self.load_img(self.data_file['train_img'])
        self.dataset['train_label']=self.load_label(self.data_file['train_label'])
        self.dataset['test_img']=self.load_img(self.data_file['test_img'])
        self.dataset['test_label']=self.load_label(self.data_file['test_label'])
        #データの保存
        self.save_file=self.dataset_dir+'/mnist.pkl'
        with open(self.save_file,'wb') as f:
            pickle.dump(self.dataset,f,-1)

        print("saved the dataset as a pickle file")


    def to_one_hot(self,label,dimention=10): #ラベルデータをone_hot形式に変える関数
        results=np.zeros((len(label), dimention))
        for i in range(label.size):
            results[i][label[i]]=1
        return results


    def normalize(self,img_data): #画像データを正規化する関数
        img_data=img_data.astype(np.float32)
        img_data/=255
        return img_data


    def preprocess_data(self): #mainファイルで実行
        with open(self.save_file, 'rb') as f:
            dataset=pickle.load(f)
        train_imgs=self.normalize(dataset['train_img'])
        test_imgs=self.normalize(dataset['test_img'])
        train_labels=self.to_one_hot(dataset['train_label'])
        test_labels=self.to_one_hot(dataset['test_label'])
        return train_imgs,train_labels,test_imgs,test_labels


    def add_noise(self,noise_rate): #mainファイルで実行 noise_rate=0~25 ランダムにノイズを加える
        train_imgs,train_labels,test_imgs,test_labels=self.preprocess_data()
        for i in range(len(train_imgs)):
            for j in range(len(train_imgs[0])):
                a=random.uniform(0,100)
                if a<=noise_rate:
                    train_imgs[i][j]=random.random()
        return train_imgs
