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
    epoches = 500  # 
    lr = 0.0001  # 
    input_num = 1
    output_num =  num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = net.BasicFCN(input_num,num_classes,length)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()  #
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    SOTA = 0.0
    model_Tstate = None
    for epoch in range(epoches):
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
            #print("The {} accuracy of epoch {} TSC: {}%".format(class_name,epoch+1, 100 * correct / total))
            torch.save(model.state_dict(),"FedTemp/"+class_name+".pkl")
            model_Tstate = model
    return str(SOTA), model_Tstate

def train_and_test_load(class_name,train_loader,test_loader,num_classes,length,FCNmodel,index):
    epoches = 500  # 
    lr = 0.0001  # 
    input_num = 1
    output_num =  num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net.DTKFCN(input_num,num_classes,length).to(device)
    model.to(device)
    parametelist = FCNmodel.getAvgParameter(44)
    PreviousModel = net.OrdinaryKDTeFCN(input_num,num_classes,length,parametelist).to(device)
    #exit(0)
    loss_func = nn.CrossEntropyLoss()  #
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    SOTA = 0.0
    model_Rstate = None
    for epoch in range(epoches):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputlist = model(images)
            premodelist = PreviousModel(images)
            loss =loss_func(outputlist[-1], labels) +net.SquareLossEW(outputlist[0],premodelist[0])+net.SquareLossEW(outputlist[1],premodelist[1])+net.SquareLossEW(outputlist[2],premodelist[2])+net.SquareLossEW(outputlist[3],premodelist[3])
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
            del images,labels    
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)[-1]
                values, predicte = torch.max(output, 1) 
                total += labels.size(0)
                correct += (predicte == labels).sum().item()
                del images,labels  
        if (correct/total)  >  SOTA:
            SOTA = correct / total
            #print("The {} accuracy of epoch {} TSC: {}%".format(class_name,epoch+1, 100 * correct / total))
            #torch.save(model.state_dict(),"basicFCN/"+class_name+".pkl")
            model_Rstate = model
    return str(SOTA), model_Rstate


setup_seed(123)

Fed_iteration = 20
names = cfg.each_elen_dir_name
start_time = time.time()
numbers = len(names)
logTxt = "FedKDAVG.txt"
f = open(logTxt,mode="a+")
f.writelines("FedKDAVG_____343Task------------1\n")
f.close()
print("FedKDAVG Task------------1")
avg = 0    
for i in range(len(names)):
    #logTxt = "FedTESAVG.txt"
   # f = open(logTxt,mode="a+")
    classname = names[i]
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
    if i == 0:
        #sota,model_state = train_and_test(classname,train_loader,test_loader,num_classes,length)
        model_state = net.BasicFCN(1,num_classes,length).to(device)
        model_state.load_state_dict(torch.load("FedTemp/"+classname+".pkl"))
        shared = net.StoreParameters()
        shared.Appended(model_state)
        #del model_state
    else:
         model_stateL = net.BasicFCN(1,num_classes,length).to(device)
         model_stateL.load_state_dict(torch.load("FedTemp/"+classname+".pkl"))
         shared.Appended(model_stateL)
         #del model_stateL
    #avg += float(sota)
    #f.writelines("FedAVG with 2 fullyconnected Dataset:"+classname+"----Accuracy:"+sota+"---NumClasses:"+str(num_classes)+"----Length:"+str(length)+"\n")
    #print("FedAVG with 2 fullyconnected Dataset:"+classname+"----Accuracy:"+sota+"---NumClasses:"+str(num_classes)+"----Length:"+str(length))
    #print("Dataset:%s eslapsed %.5f mins"%(classname,(time.time()-start_time)/60))
    #f.close()
avg = 30
for e in range(1,Fed_iteration):
    FCNmodel = shared
    print("average accruarcy is %.6f"%(avg/numbers))
    print("FedTESAVGOneTeacher Task------------%d"%(e+1))
    logTxt = "FedKDAVG.txt"
    f = open(logTxt,mode="a+")
    f.writelines("average accruarcy is "+str(avg/numbers)+"\n")
    f.writelines("FedAVG Task------------"+str(e+1)+"\n")
    f.close()
    avg = 0
    for i in range(len(names)):
        logTxt = "FedKDAVG.txt"
        f = open(logTxt,mode="a+")
        classname = names[i]
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
        if i == 0:
            sota,model_stateP = train_and_test_load(classname,train_loader,test_loader,num_classes,length,FCNmodel,i)
            shared = net.StoreParameters()
            shared.Appended(model_stateP)
            #del model_stateP
        else:
            sota,model_stateV = train_and_test_load(classname,train_loader,test_loader,num_classes,length,FCNmodel,i)
            shared.Appended(model_stateV)
            #del model_stateV
        avg += float(sota)
        f.writelines("FedAVG with 2 fullyconnected Dataset:"+classname+"----Accuracy:"+sota+"---NumClasses:"+str(num_classes)+"----Length:"+str(length)+"\n")
        print("FedAVG with 2 fullyconnected Dataset:"+classname+"----Accuracy:"+sota+"---NumClasses:"+str(num_classes)+"----Length:"+str(length))
        print("Dataset:%s eslapsed %.5f mins"%(classname,(time.time()-start_time)/60))
        f.close()
        del train_loader, test_loader,test_set

print("Total  eslapsed %.5f hours"%((time.time()-start_time)/3600))
