import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Subset,Dataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import network as net
from random import  shuffle
import random
import data
import os
import config as cfg 
import time
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class LoadData(Dataset):
    def __init__(self,train_x,train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.len = len(self.train_x)

    def __getitem__(self,index):
        return self.train_x[index],self.train_y[index]

    def __len__(self):
        return self.len

def train_and_test(class_name,train_loader,test_loader,num_classes,length):
    epoches = 800  # 
    lr = 0.0001  # 
    input_num = 1
    output_num =  num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = net.BasicFCN(input_num,num_classes,length)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()  #
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    SOTA = 0.1
    for epoch in range(epoches):
        flag = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()  
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                values, predicte = torch.max(output, 1) 
                total += labels.size(0)
                correct += (predicte == labels).sum().item()
        if (correct/total)  >  SOTA:
            SOTA = correct / total
            print("The {} accuracy of epoch {} TSC: {}%".format(class_name,epoch+1, 100 * correct / total))
            torch.save(model.state_dict(),"basicFCN/"+class_name+".pkl")
    return str(SOTA)

setup_seed(123)


names = cfg.sub_dir_name
start_time = time.time()
for name in names:
    logTxt = "BasicFCNLog.txt"
    f = open(logTxt,mode="a+")
    classname = name
    x_train = np.load("data/"+classname+"_Xtrain.npy")
    y_train = np.load("data/"+classname+"_Ytrain.npy")
    x_test = np.load("data/"+classname+"_Xtest.npy")
    y_test = np.load("data/"+classname+"_Ytest.npy")
    num_classes = y_test.shape[1]
    length = x_train.shape[1]
    ###
    y_test = np.argmax(y_test,axis=1)
    y_train =  np.argmax(y_train,axis=1)
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1])).astype(np.float32)
    x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1])).astype(np.float32)
    train_loader = LoadData(x_train,y_train)
    test_set = LoadData(x_test,y_test)
    train_loader = dataloader.DataLoader(dataset=train_loader,batch_size=x_train.shape[0]//4,shuffle=True)
    test_loader = dataloader.DataLoader(dataset=test_set,shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Load---------dataset:",classname)
    sota = train_and_test(classname,train_loader,test_loader,num_classes,length)
    f.writelines("FCN with 2 fullyconnected Dataset:"+classname+"----Accuracy:"+sota+"---NumClasses:"+str(num_classes)+"----Length:"+str(length)+"\n")
    print("Dataset:%s eslapsed %.5f mins"%(classname,(time.time()-start_time)/60))
    f.close()
print("Total  eslapsed %.5f hours"%((time.time()-start_time)/3600))






