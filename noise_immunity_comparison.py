import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from data_preprocessing import *
from neural_network import *
from optimizer import *
from main import Training


def compare():
    noise_per_list=[i for i in range(26)]
    data=DataPreprocessing()
    acc_array=np.zeros((6,26))

    for i in range(6): #train_noise 0,5,10,15,20,25%を比較
        train_imgs,train_labels,test_imgs,test_labels=data.preprocess_data()
        if i!=0:
            train_imgs=np.load("dataset/noise_added/train/train_img_noise"+str(5*i)+"%.npy")

        for j in range(26):
            if j!=0:
                test_imgs=np.load("dataset/noise_added/test/test_img_noise"+str(j)+"%.npy")
            network=NeuralNetWork()
            opt=SGD(learning_rate=0.01)
            training=Training(network=network, optimizer=opt, train_imgs=train_imgs, train_labels=train_labels, test_imgs=test_imgs, test_labels=test_labels, dataset_loops=50, batch_size=100) #dataset_loops変更可 50くらいでも良さそう
            training.startTraining()
            acc_array[i][j]=training.test_accuracy_list[-1]
            #save_dir="acc_arrays/acc_array"+str(5*i)+".npy"
            #np.save(save_dir,acc_array[i])

            print("Finish train_noise: {}%, test_noise: {}%".format(5*i,j))

    save_dir="noise_immunity_comparison.npy"

    np.save(save_dir,acc_array)

    # 描画
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    ax.plot(noise_per_list, acc_array[0], color='blue', label="train noise 0%")
    ax.plot(noise_per_list, acc_array[1], color='green', label="train noise 5%")
    ax.plot(noise_per_list, acc_array[2], color='red', label="train noise 10%")
    ax.plot(noise_per_list, acc_array[3], color='cyan', label="train noise 15%")
    ax.plot(noise_per_list, acc_array[4], color='pink', label="train noise 20%")
    ax.plot(noise_per_list, acc_array[5], color='yellow', label="train noise 25%")

    ax.set_title('comparison of accuracy with noise')
    ax.set_xlabel('test noise [%]')
    ax.set_ylabel('accuracy')
    ax.grid(True)
    ax.legend()
    fig.show()

    SAVE_DIR="result_graphs/comparison_with_noise.png"
    plt.savefig(SAVE_DIR)


if __name__ == "__main__":
    compare()
