from data_preprocessing import *
import numpy as np


def make_noise():
    width=28
    height=28
    data=DataPreprocessing()
    train_imgs,train_labels,test_imgs,test_labels=data.preprocess_data()
    data_size=width*height

    for i in range(1,26):
        noise_num=int(0.01*i*data_size)
        train_noise_imgs=train_imgs.copy()

        for each_img in train_noise_imgs:
            noise_filter=np.random.randint(0,data_size,noise_num)

            for noise_idx in noise_filter:
                each_img[noise_idx]=np.random.random()

        save_dir="dataset/noise_added/train/train_img_noise"+str(i)+"%.npy"

        np.save(save_dir,train_noise_imgs)
        print("train noise {}% is written.".format(i))


    for j in range(1,26):
        noise_num=int(0.01*j*data_size)
        test_noise_imgs=test_imgs.copy()

        for each_img in test_noise_imgs:
            noise_filter=np.random.randint(0,data_size,noise_num)

            for noise_idx in noise_filter:
                each_img[noise_idx]=np.random.random()

        save_dir="dataset/noise_added/test/test_img_noise"+str(j)+"%.npy"

        np.save(save_dir,test_noise_imgs)
        print("test noise {}% is written.".format(j))


if __name__ == "__main__":
    make_noise()
