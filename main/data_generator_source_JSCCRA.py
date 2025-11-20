from random import sample
import random
import numpy as np
import keras
from keras.utils import np_utils
# get the training sampels
def train_datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,train_list, channel):
    while True:
        x_train, y_train = list(range(batchsize)), list(range(batchsize))
        index_list = list(range(batchsize))
        target_list = list(range(40))
        sub_list = list(range(len(train_data1)))

        # parameter alpha of mixstyle
        alpha = 0.1
        # parameter K_max of mpcr
        K_mak = 7
        # the j-th smaple list index
        list_batchsize = list(range(batchsize))
        random.shuffle(list_batchsize)

        for i in range(int(batchsize)):
            #save mix ratio drawn from beta distribution
            index_list[i] = np.random.beta(alpha,alpha)
            m = sample(target_list, 1)[0]
            k = sample(train_list, 1)[0]
            s = sample(sub_list, 1)[0]
            # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time
            time_start = random.randint(35+125,int(1250+35+125-win_train))
            time_end = time_start + win_train
            # get four sub-inputs
            x_11 = train_data1[s][k][m][:,time_start:time_end]
            x_21 = np.reshape(x_11,(channel, win_train, 1))
            x_12 = train_data2[s][k][m][:,time_start:time_end]
            x_22 = np.reshape(x_12,(channel, win_train, 1))
            x_13 = train_data3[s][k][m][:,time_start:time_end]
            x_23 = np.reshape(x_13,(channel, win_train, 1))

            # concatenate the four sub-input into one input
            x_concatenate = np.concatenate((x_21, x_22, x_23), axis=-1)
            # save the training sample and corresponding label
            x_train[i] = x_concatenate
            y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')

        ### style augmentation
        # reshape the mixing ration list
        index_style = np.reshape(index_list,(batchsize,1,1))
        # obtain original training samples
        x_original = np.array(x_train)
        # reshape training samples to 2-D size facilitate the application of RCC
        x_trian_flatten = np.reshape(x_original,(batchsize,channel, win_train*3))
        # get the i-th channle-wise mean list
        mean_x_o = np.mean(x_trian_flatten,axis = 2)
        # reshape the i-th channle-wise mean list
        mean_x_o = np.reshape(mean_x_o,(batchsize,channel, 1))
        # get the j-th channle-wise mean list
        mean_x_g = mean_x_o[list_batchsize,:,:]
        # get the i-th channle-wise std list
        std_x_o = np.std(x_trian_flatten,axis = 2)
        # reshape the i-th channle-wise std list
        std_x_o = np.reshape(std_x_o,(batchsize,channel, 1))
        # get the j-th channle-wise std list
        std_x_g = std_x_o[list_batchsize,:,:]
        # get i-th mixed channel-wise mean
        mean_mix = index_style*mean_x_o + (1-index_style)*mean_x_g
        # get i-th mixed channel-wise std
        std_mix = index_style*std_x_o + (1-index_style)*std_x_g
        # get the standardized sample list 
        x_norm_list = (x_trian_flatten-mean_x_o)/std_x_o
        # y_train1 = np.reshape(y_train,(batchsize,40))

        ### content augmentation
        recon_list = list(range(batchsize))
        for i in range(batchsize):
            # randomly selecting a k value (n_channel) from the range [1, 6]
            n_channel = sample(list(range(1,K_mak)), 1)[0]
            # randomly selecting n_channel channels from the 9 channels of a training sample
            channel_n = sample(list(range(9)), n_channel)#[0]
            # get the i-th standardized sample 
            x_data_norm = x_norm_list[i]
            # obtaining the covariance matrix
            x_conv1 = np.dot(x_data_norm,x_data_norm.T)/(win_train*3-1)
            # eigenvalues and eigenvectors calculation
            _, featVec1=np.linalg.eig(x_conv1)
            # principal component representation calculation
            dere_x = np.dot(featVec1.T,x_data_norm) #np.linalg.inv(featVec1),featVec1.T
            # masked principal component representation calculation
            dere_x[channel_n,:] = 0
            # data reconstruction
            recon_x = np.dot(featVec1,dere_x)
            
            recon_list[i] = recon_x
        x_recon = np.array(recon_list)

        ### fusing the augmented style and content
        x_mix = std_mix*x_recon + mean_mix
        # # reshape training samples to 3-D size as the model input
        x_out = np.reshape(x_mix,(batchsize,channel, win_train, 3))
        y_out = np.reshape(y_train,(batchsize,40))
         

        yield x_out, y_out

