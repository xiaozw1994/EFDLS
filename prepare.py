import numpy as np
import os 
import data
import config as cfg
import keras.utils as utils


dirs = cfg.data_files
for v in os.listdir(dirs):
    class_name = v
    subdir = os.path.join(dirs,v)
    train_file_name = os.path.join(subdir,v+"_TRAIN")
    test_file_name =  os.path.join(subdir,v+"_TEST")
    x_train,y_train = data.readucr(train_file_name)
    x_test,y_test = data.readucr(test_file_name)
    num_classes = len(np.unique(y_test))
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(num_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(num_classes-1)
    Y_train = utils.np_utils.to_categorical(y_train, num_classes)
    Y_test = utils.np_utils.to_categorical(y_test, num_classes)
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean)/(x_train_std)
    x_test = (x_test - x_test.mean(axis=0))/(x_test.std(axis=0))
    print(class_name,"--------------------",x_train.shape,Y_train.shape,x_test.shape,Y_test.shape)
    np.save("data/"+class_name+"_Xtrain",x_train)
    np.save("data/"+class_name+"_Ytrain",Y_train)
    np.save("data/"+class_name+"_Xtest",x_test)
    np.save("data/"+class_name+"_Ytest",Y_test)