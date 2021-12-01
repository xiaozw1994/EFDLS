import config as cfg
import numpy as np
import os 
import matplotlib.pyplot as plt
from keras.utils import  np_utils
import tensorflow as tf
import random
#
#
# reading data from the txt files
#
def readucr(filename):
    data = np.loadtxt(filename+".tsv", delimiter = '\t')
    Y = data[:,0]
    X = data[:,1:]
    if np.isnan(X).any():
        X[np.isnan(X)] = np.nanmean(X)
    return X, Y
###
#    
#  To normalize the trained lebeled data
def NormalizationClassification(Y,num_classes):
    Y = np.array(Y)
    return (Y-Y.mean()) / (Y.max()-Y.mean()) *(num_classes-1)
#
#
#
def NormalizationFeatures(X):
    X = np.array(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    value = (X-mean) / std
    return value
#
#  This file is used to get the size of class in the training dataset
#
def GetNumClasses(y):
    y = np.array(y)
    num_classes = len(np.unique(y))
    return num_classes
#### Noising
#   Using the Gussian function 
#
def Noising(x,loc=cfg.loc,scale=cfg.scale):
    x = np.array(x)
    shape = x.shape
    x = np.random.normal( loc=loc,scale=scale,size=shape) + x
    return x
#
#   To OneHot
#
def OneHot(y,num_classes):
    y = np.array(y)
    y = np_utils.to_categorical(y,num_classes)
    return y
#
#
#Show The index of picure
#
def Show(train_x,aug_x,index,length):
    x = [i for i in range(1,length+1)]
    fig = plt.figure()
    aix = fig.subplots(nrows=2,ncols=1)
    aix[0].plot(x,train_x[index])
    aix[1].plot(x,aug_x[index])
    plt.show()
#
#
# Agumentation 
#
def Augmentation(train_x):
    x_shape = train_x.shape
    list_len = len(cfg.locslist)
    new_train = np.array(np.zeros((x_shape[0]*(list_len),x_shape[1])))
    for i in range(list_len):
        loc = cfg.locslist[i]
        scale = cfg.scalelist[i]
        new_train[i*x_shape[0]:(i+1)*x_shape[0],...] = np.random.normal( loc=loc,scale=scale,size=x_shape) + train_x
    return new_train 

def showRand(train_x,length):
    index = np.random.randint(0,length)
    l = 6
    x = [i for i in range(1,train_x.shape[1]+1)]
    fig = plt.figure()
    aix = fig.subplots(nrows=l,ncols=1)
    for i in range(0,l):
        aix[i].plot(x,train_x[i*length+index,...])
    plt.show()

############################
########## Load Files
############################
def SplitDataset(train_x,train_y,num_classes):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    indices = np.random.permutation(train_x.shape[0])
    train_x , train_y = train_x[indices],train_y[indices]
    perclass = cfg.per_class[2]
    labeled_feature = []
    labeled_y = []
    unlabeled = []
    for i in range(1,num_classes+1):
        tmp_index = np.where(train_y==i)
        tmp_train = train_x[tmp_index]
        tmp_label = train_y[tmp_index]
        tmp_len = len(tmp_train)
        if tmp_len > perclass:
            tmp_labeled = tmp_train[:perclass,...]
            tmp_y = tmp_label[:perclass,...]
            tmp_unlabel = tmp_train[perclass:,...]
            #print(tmp_unlabel.shape)
            labeled_feature.append(tmp_labeled)
            labeled_y.append(tmp_y)
            unlabeled.append(tmp_unlabel)
        else :
            labeled_feature.append(tmp_train)
            labeled_y.append(tmp_label)
    labeled_feature = np.concatenate(labeled_feature,axis=0)
    labeled_y = np.concatenate(labeled_y,axis=0)
    unlabeled = np.concatenate(unlabeled,axis=0)
    return labeled_feature,labeled_y,unlabeled
#########################################
#  Time Series Cutout            

def TimeCutout(feature,pad_h,replace=0):
    '''
    args:
        feature : the raw data. [length,1]
        pad_h : padding size
    '''
    length = tf.shape(feature)[0]
    coutout_center_length  = tf.random_uniform(shape=[],minval=0,maxval=length,dtype=tf.int32)
    lower_pad = tf.maximum(0,coutout_center_length-pad_h)
    uperr_pad = tf.maximum(0,length-coutout_center_length-pad_h)
    cutout_shape = [
        length-(lower_pad+uperr_pad),1
    ]
    padding_dims = [[lower_pad,uperr_pad],[0,0]]
    mask = tf.pad(
        tf.zeros(cutout_shape,dtype=feature.dtype),
        padding_dims,constant_values=1
    )
    feature = tf.where(
        tf.equal(mask,0),tf.ones_like(feature,dtype=feature.dtype)*replace,feature
    )
    return feature
##############################
######################################################
def TimeCutoutNumpy(feature,pad,replace=0):
    '''
    args:
        feature : the raw data. [length,1]
        pad : padding size
    '''
    length = feature.shape[0]
    coutout_center_length = np.random.randint(low=0,high=length)
    lower_padd = np.maximum(0,coutout_center_length-pad)
    upper_padd = np.maximum(0,length-coutout_center_length-pad)
    cutout_shape = [
        length-(lower_padd+upper_padd),1
    ]
    pad_dims = [[lower_padd,upper_padd],[0,0]]
    mask = np.pad(
        np.zeros(cutout_shape,dtype=feature.dtype),
        pad_dims,'constant',constant_values=1
    )
    feature = np.where(
        np.equal(mask,0),np.ones_like(feature,dtype=feature.dtype)*replace,feature
    )
    return feature

def TimeCutoutSingle(feature,pad,cutout,replace=0):
    '''
    args:
        feature : the raw data. [length,1]
        pad : padding size
    '''
    length = feature.shape[0]
    coutout_center_length = cutout
    lower_padd = np.maximum(0,coutout_center_length-pad)
    upper_padd = np.maximum(0,length-coutout_center_length-pad)
    cutout_shape = [
        length-(lower_padd+upper_padd),1
    ]
    pad_dims = [[lower_padd,upper_padd],[0,0]]
    mask = np.pad(
        np.zeros(cutout_shape,dtype=feature.dtype),
        pad_dims,'constant',constant_values=1
    )
    feature = np.where(
        np.equal(mask,0),np.ones_like(feature,dtype=feature.dtype)*replace,feature
    )
    return feature

####################### Policy################
def Policy(Batch,length):
    ###
    ##
    '''
    Batch : represents the number of sample
    length: the length of  time series dataset
    '''
    highlength = Batch//2
    lowlength = Batch - highlength
    highrate = np.random.uniform(low=0.5,high=1.0,size=1)
    lowrate = np.random.uniform(low=0.0,high=0.5,size=1)
    decay = 0.99
    rates = []
    for i in range(highlength):
        highrate = highrate * decay + (1-highrate) * (1-decay)
        if highrate < 0.4:
            highrate =  np.random.uniform(low=0.5,high=1.0,size=1)
        else :
            rates.append(highrate[0])
    for i in range(lowlength):
        lowrate = lowrate * decay  
        if lowrate < 0.0:
            lowrate = np.random.uniform(low=0.0,high=0.5,size=1)
        else :
            rates.append(lowrate[0])
    rates = [int(i*length) for i in rates]
    return rates
def AutoAugmentation(unlabeled):
    ###############
    ####
    batch = unlabeled.shape[0]
    length = unlabeled.shape[1]
    unlabeled = unlabeled.reshape((batch,length,1))
    rates = Policy(batch,length)
    feat = []
    for i in range(len(rates)):
        rate = rates[i]
        unlabel = unlabeled[i]
        pad = np.random.randint(0,5,1)[0]
        feature = TimeCutoutSingle(unlabel,pad,rate)
        feat.append(feature)
    feat = np.array(feat).reshape((-1,length))
    return feat
######################Augmentation Operations##########################################
def SemiAugmentation(unlabeled,k=1):
    if k==1:
        raw_data = unlabeled
        auto = AutoAugmentation(unlabeled)
    else :
        raw_data = []
        auto = []
        raw_data.append(unlabeled)
        for i in range(k):
            au = AutoAugmentation(unlabeled)
            auto.append(au)  
            if i >0  :
                para = np.random.uniform(low=0.05,high=0.2,size=2)
                feat =  np.random.normal(loc=para[0],scale=para[1],size=unlabeled.shape)+unlabeled
                raw_data.append(feat)
        raw_data,auto = np.concatenate(raw_data,0),np.concatenate(auto,0)
    return raw_data,auto