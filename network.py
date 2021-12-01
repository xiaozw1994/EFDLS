import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Subset,Dataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
from random import  shuffle

class BasicFCN(nn.Module):
    def __init__(self,input_dims,num_classes,length):
        super(BasicFCN,self).__init__()
        self.conv1 = nn.Conv1d(input_dims,128,9,stride=1,padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 =nn.Conv1d(128,256,5,1,padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256,128,5,1,2)
        self.bn3 = nn.BatchNorm1d(128)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.ReLU = nn.ReLU()
        self.linear1 = nn.Linear(128,128)
        self.linear2 = nn.Linear(128,num_classes)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.bn1.weight,val=1)
        nn.init.constant_(self.bn2.weight,1)
        nn.init.constant_(self.bn3.weight,val=1)
        nn.init.constant_(self.conv1.bias,val=0)
        nn.init.constant_(self.bn1.bias,val=0)
        nn.init.constant_(self.conv2.bias,val=0)
        nn.init.constant_(self.bn2.bias,val=0)
        nn.init.constant_(self.conv3.bias,val=0)
        nn.init.constant_(self.bn3.bias,val=0)
        nn.init.constant_(self.linear1.bias,val=0)
        nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.ReLU(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.ReLU(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.ReLU(bn3)
        avg = self.avg(relu3)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(drop)
        return res

###########Create Basic Shared Class
'''
conv1.weight
conv1.bias
bn1.weight
bn1.bias
bn1.running_mean
bn1.running_var
bn1.num_batches_tracked
conv2.weight
conv2.bias
bn2.weight
bn2.bias
bn2.running_mean
bn2.running_var
bn2.num_batches_tracked
conv3.weight
conv3.bias
bn3.weight
bn3.bias
bn3.running_mean
bn3.running_var
bn3.num_batches_tracked
linear1.weight
linear1.bias
linear2.weight
linear2.bias
'''
class BasicShared(object):
    def __init__(self,model):
        self.conv1_weight = model.conv1.weight.data
        self.conv1_bias = model.conv1.bias.data
        self.bn1_weight = model.bn1.weight.data
        self.bn1_bias = model.bn1.bias.data
        self.conv2_weight = model.conv2.weight.data
        self.conv2_bias = model.conv2.bias.data
        self.bn2_weight = model.bn2.weight.data
        self.bn2_bias = model.bn2.bias.data
        self.conv3_weight = model.conv3.weight.data
        self.conv3_bias = model.conv3.bias.data
        self.bn3_weight = model.bn3.weight.data
        self.bn3_bias = model.bn3.bias.data
        self.linea1_weight = model.linear1.weight.data
        self.linear1_bias = model.linear1.bias.data
    def add(self,model):
        self.conv1_weight  +=  model.conv1.weight.data
        self.conv1_bias +=  model.conv1.bias.data
        self.bn1_weight +=  model.bn1.weight.data
        self.bn1_bias  +=  model.bn1.bias.data
        self.conv2_weight  +=  model.conv2.weight.data
        self.conv2_bias  +=  model.conv2.bias.data
        self.bn2_weight +=  model.bn2.weight.data
        self.bn2_bias  +=  model.bn2.bias.data
        self.conv3_weight  +=  model.conv3.weight.data
        self.conv3_bias  +=  model.conv3.bias.data
        self.bn3_weight  +=  model.bn3.weight.data
        self.bn3_bias  +=  model.bn3.bias.data
        self.linea1_weight  +=  model.linear1.weight.data
        self.linear1_bias  +=  model.linear1.bias.data
    def dividition(self,integrates):
        self.conv1_weight  /=  integrates
        self.conv1_bias  /=  integrates
        self.bn1_weight  /=  integrates
        self.bn1_bias   /=  integrates
        self.conv2_weight  /=  integrates
        self.conv2_bias   /=  integrates
        self.bn2_weight  /=  integrates
        self.bn2_bias   /=  integrates
        self.conv3_weight   /=  integrates
        self.conv3_bias   /=  integrates
        self.bn3_weight   /=  integrates
        self.bn3_bias   /=  integrates
        self.linea1_weight /= integrates
        self.linear1_bias  /= integrates
###
#############
class LoadBasicFCN(nn.Module):
    def __init__(self,input_dims,num_classes,basicModel):
        super(LoadBasicFCN,self).__init__()
        self.conv1 = nn.Conv1d(input_dims,128,9,stride=1,padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 =nn.Conv1d(128,256,5,1,padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256,128,5,1,2)
        self.bn3 = nn.BatchNorm1d(128)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.ReLU = nn.ReLU()
        self.linear1 = nn.Linear(128,128)
        self.linear2 = nn.Linear(128,num_classes)
        self.conv1.weight = Parameter(basicModel.conv1_weight)
        self.conv1.bias = Parameter(basicModel.conv1_bias)
        self.bn1.weight = Parameter(basicModel.bn1_weight)
        self.bn1.bias = Parameter(basicModel.bn1_bias)
        self.conv2.weight = Parameter(basicModel.conv2_weight)
        self.conv2.bias = Parameter(basicModel.conv2_bias)
        self.bn2.weight = Parameter(basicModel.bn2_weight)
        self.bn2.bias = Parameter(basicModel.bn2_bias)
        self.conv3.weight = Parameter(basicModel.conv3_weight)
        self.conv3.bias = Parameter(basicModel.conv3_bias)
        self.bn3.weight = Parameter(basicModel.bn3_weight)
        self.bn3.bias = Parameter(basicModel.bn3_bias)
        self.linear1.weight = Parameter(basicModel.linea1_weight)
        self.linear1.bias = Parameter(basicModel.linear1_bias)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.ReLU(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.ReLU(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.ReLU(bn3)
        avg = self.avg(relu3)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(drop)
        return res

def SquareLossEW(feature_target,feature_source):
    loss = torch.mean(torch.pow((feature_target- feature_source), 2))
    return loss

def GetWCLoss(Previsous,Current):
    conv1_loss = SquareLossEW(Previsous.conv1.weight,Current.conv1.weight) + SquareLossEW(Previsous.conv1.bias,Current.conv1.bias)
    bn1_loss = SquareLossEW(Previsous.bn1.weight,Current.bn1.weight) + SquareLossEW(Previsous.bn1.bias,Current.bn1.bias)
    conv2_loss = SquareLossEW(Previsous.conv2.weight,Current.conv2.weight) + SquareLossEW(Previsous.conv2.bias,Current.conv2.bias)
    bn2_loss = SquareLossEW(Previsous.bn2.weight,Current.bn2.weight) + SquareLossEW(Previsous.bn2.bias,Current.bn2.bias)
    conv3_loss = SquareLossEW(Previsous.conv3.weight,Current.conv3.weight) + SquareLossEW(Previsous.conv3.bias,Current.conv3.bias)
    bn3_loss = SquareLossEW(Previsous.bn3.weight,Current.bn3.weight) + SquareLossEW(Previsous.bn3.bias,Current.bn3.bias)
    linear1_loss = SquareLossEW(Previsous.linear1.weight,Current.linear1.weight) + SquareLossEW(Previsous.linear1.bias,Current.linear1.bias)
    WCloss = conv1_loss + conv2_loss + conv3_loss + bn1_loss+bn2_loss+bn3_loss+linear1_loss
    return WCloss

#######################Load Ordinal Transfer Learning

class OrdinaryTransferFCN(nn.Module):
    def __init__(self,input_dims,num_classes,length,modelFCN):
        super(OrdinaryTransferFCN,self).__init__()
        self.conv1 = nn.Conv1d(input_dims,128,9,stride=1,padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 =nn.Conv1d(128,256,5,1,padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256,128,5,1,2)
        self.bn3 = nn.BatchNorm1d(128)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.ReLU = nn.ReLU()
        self.linear1 = nn.Linear(128,128)
        self.linear2 = nn.Linear(128,num_classes)
        self.conv1.weight = modelFCN.conv1.weight
        self.conv2.weight = modelFCN.conv2.weight
        self.conv3.weight = modelFCN.conv3.weight
        self.linear1.weight = modelFCN.linear1.weight
        nn.init.kaiming_normal_(self.linear2.weight)
        self.bn1.weight = modelFCN.bn1.weight
        self.bn2.weight = modelFCN.bn2.weight
        self.bn3.weight = modelFCN.bn3.weight
        self.conv1.bias = modelFCN.conv1.bias
        self.bn1.bias = modelFCN.bn1.bias
        self.conv2.bias = modelFCN.conv2.bias
        self.bn2.bias = modelFCN.bn2.bias
        self.conv3.bias = modelFCN.conv3.bias
        self.bn3.bias = modelFCN.bn3.bias
        self.linear1.bias = modelFCN.linear1.bias
        nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.ReLU(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.ReLU(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.ReLU(bn3)
        avg = self.avg(relu3)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(drop)
        return res
#########################################################
###################DTK
####################################################

class DTKLIST(nn.Module):
    def __init__(self,input_dims,num_classes,length,modelFCN):
        super(DTKLIST,self).__init__()
        self.conv1 = nn.Conv1d(input_dims,128,9,stride=1,padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 =nn.Conv1d(128,256,5,1,padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256,128,5,1,2)
        self.bn3 = nn.BatchNorm1d(128)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.ReLU = nn.ReLU()
        self.linear1 = nn.Linear(128,128)
        self.linear2 = nn.Linear(128,num_classes)
        self.conv1.weight = modelFCN.conv1.weight
        self.conv2.weight = modelFCN.conv2.weight
        self.conv3.weight = modelFCN.conv3.weight
        self.linear1.weight = modelFCN.linear1.weight
        nn.init.kaiming_normal_(self.linear2.weight)
        self.bn1.weight = modelFCN.bn1.weight
        self.bn2.weight = modelFCN.bn2.weight
        self.bn3.weight = modelFCN.bn3.weight
        self.conv1.bias = modelFCN.conv1.bias
        self.bn1.bias = modelFCN.bn1.bias
        self.conv2.bias = modelFCN.conv2.bias
        self.bn2.bias = modelFCN.bn2.bias
        self.conv3.bias = modelFCN.conv3.bias
        self.bn3.bias = modelFCN.bn3.bias
        self.linear1.bias = modelFCN.linear1.bias
        nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        reslist = []
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.ReLU(bn1)
        reslist.append(relu1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.ReLU(bn2)
        reslist.append(relu2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.ReLU(bn3)
        avg = self.avg(relu3)
        reslist.append(relu3)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        reslist.append(ful1)
        #drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(ful1)
        reslist.append(res)
        return reslist
###########################################################

class DTKFCN(nn.Module):
    def __init__(self,input_dims,num_classes,length):
        super(DTKFCN,self).__init__()
        self.conv1 = nn.Conv1d(input_dims,128,9,stride=1,padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 =nn.Conv1d(128,256,5,1,padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256,128,5,1,2)
        self.bn3 = nn.BatchNorm1d(128)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.ReLU = nn.ReLU()
        self.linear1 = nn.Linear(128,128)
        self.linear2 = nn.Linear(128,num_classes)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.bn1.weight,val=1)
        nn.init.constant_(self.bn2.weight,1)
        nn.init.constant_(self.bn3.weight,val=1)
        nn.init.constant_(self.conv1.bias,val=0)
        nn.init.constant_(self.bn1.bias,val=0)
        nn.init.constant_(self.conv2.bias,val=0)
        nn.init.constant_(self.bn2.bias,val=0)
        nn.init.constant_(self.conv3.bias,val=0)
        nn.init.constant_(self.bn3.bias,val=0)
        nn.init.constant_(self.linear1.bias,val=0)
        nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        reslist = []
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.ReLU(bn1)
        reslist.append(relu1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.ReLU(bn2)
        reslist.append(relu2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.ReLU(bn3)
        avg = self.avg(relu3)
        reslist.append(relu3)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        reslist.append(ful1)
        #drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(ful1)
        reslist.append(res)
        return reslist
##################################################################

class OrdinaryTTeFCN(nn.Module):
    def __init__(self,input_dims,num_classes,length,modelFCN):
        super(OrdinaryTTeFCN,self).__init__()
        self.conv1 = nn.Conv1d(input_dims,128,9,stride=1,padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 =nn.Conv1d(128,256,5,1,padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256,128,5,1,2)
        self.bn3 = nn.BatchNorm1d(128)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.ReLU = nn.ReLU()
        self.linear1 = nn.Linear(128,128)
        self.linear2 = nn.Linear(128,num_classes)
        self.conv1.weight = Parameter( modelFCN[0].data)
        self.conv2.weight = Parameter(modelFCN[4].data)
        self.conv3.weight = Parameter(modelFCN[8].data)
        self.linear1.weight = Parameter(modelFCN[12].data)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.bn1.weight = Parameter(modelFCN[2].data)
        self.bn2.weight = Parameter(modelFCN[6].data)
        self.bn3.weight = Parameter(modelFCN[10].data)
        self.conv1.bias = Parameter(modelFCN[1].data)
        self.bn1.bias = Parameter(modelFCN[3].data)
        self.conv2.bias = Parameter(modelFCN[5].data)
        self.bn2.bias = Parameter(modelFCN[7])
        self.conv3.bias = Parameter(modelFCN[9])
        self.bn3.bias = Parameter(modelFCN[11].data)
        self.linear1.bias = Parameter(modelFCN[13].data)
        nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        #reslist = []
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.ReLU(bn1)
        #reslist.append(relu1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.ReLU(bn2)
        #reslist.append(relu2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.ReLU(bn3)
        avg = self.avg(relu3)
        #reslist.append(relu3)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        #reslist.append(ful1)
        drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(drop)
        return res

class OrdinaryKDTeFCN(nn.Module):
    def __init__(self,input_dims,num_classes,length,modelFCN):
        super(OrdinaryKDTeFCN,self).__init__()
        self.conv1 = nn.Conv1d(input_dims,128,9,stride=1,padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 =nn.Conv1d(128,256,5,1,padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256,128,5,1,2)
        self.bn3 = nn.BatchNorm1d(128)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.ReLU = nn.ReLU()
        self.linear1 = nn.Linear(128,128)
        self.linear2 = nn.Linear(128,num_classes)
        self.conv1.weight = Parameter( modelFCN[0].data)
        self.conv2.weight = Parameter(modelFCN[4].data)
        self.conv3.weight = Parameter(modelFCN[8].data)
        self.linear1.weight = Parameter(modelFCN[12].data)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.bn1.weight = Parameter(modelFCN[2].data)
        self.bn2.weight = Parameter(modelFCN[6].data)
        self.bn3.weight = Parameter(modelFCN[10].data)
        self.conv1.bias = Parameter(modelFCN[1].data)
        self.bn1.bias = Parameter(modelFCN[3].data)
        self.conv2.bias = Parameter(modelFCN[5].data)
        self.bn2.bias = Parameter(modelFCN[7])
        self.conv3.bias = Parameter(modelFCN[9])
        self.bn3.bias = Parameter(modelFCN[11].data)
        self.linear1.bias = Parameter(modelFCN[13].data)
        nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        reslist = []
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.ReLU(bn1)
        reslist.append(relu1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.ReLU(bn2)
        reslist.append(relu2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.ReLU(bn3)
        avg = self.avg(relu3)
        reslist.append(relu3)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        reslist.append(ful1)
        #drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(ful1)
        return reslist



####################################################
############# FedNews
#####################################################
class StoreParameters(object):
    def __init__(self):
        self.conv1_weight = []
        self.conv1_bias = []
        self.bn1_weight= []
        self.bn1_bias = []
        self.conv2_weight = []
        self.conv2_bias = []
        self.bn2_weight = []
        self.bn2_bias= []
        self.conv3_weight = []
        self.conv3_bias = []
        self.bn3_weight = []
        self.bn3_bias= []
        self.linea1_weight = []
        self.linear1_bias= []
        self.strores = []
    def Appended(self,model):
        self.conv1_weight.append(model.conv1.weight)
        self.conv1_bias.append(model.conv1.bias)
        self.bn1_weight.append(model.bn1.weight)
        self.bn1_bias.append( model.bn1.bias)
        self.conv2_weight.append(model.conv2.weight)
        self.conv2_bias.append(model.conv2.bias)
        self.bn2_weight.append( model.bn2.weight)
        self.bn2_bias.append( model.bn2.bias)
        self.conv3_weight.append( model.conv3.weight)
        self.conv3_bias .append( model.conv3.bias)
        self.bn3_weight.append( model.bn3.weight)
        self.bn3_bias.append( model.bn3.bias)
        self.linea1_weight.append( model.linear1.weight)
        self.linear1_bias.append( model.linear1.bias)
    def ComputeEWS(self,model):
        EWSLoss = 0
        for i in range(len(self.conv1_weight)):
            conv_1_loss = SquareLossEW(self.conv1_weight[i], model.conv1.weight) + SquareLossEW(self.conv1_bias[i],model.conv1.bias)+ SquareLossEW(self.bn1_weight[i],model.bn1.weight)+SquareLossEW(self.bn1_bias[i],model.bn1.bias)
            conv2_loss = SquareLossEW(self.conv2_weight[i], model.conv2.weight) + SquareLossEW(self.conv2_bias[i],model.conv2.bias)+ SquareLossEW(self.bn2_weight[i],model.bn2.weight)+SquareLossEW(self.bn2_bias[i],model.bn2.bias)
            conv3_loss = SquareLossEW(self.conv3_weight[i], model.conv3.weight) + SquareLossEW(self.conv3_bias[i],model.conv3.bias)+ SquareLossEW(self.bn3_weight[i],model.bn3.weight)+SquareLossEW(self.bn3_bias[i],model.bn3.bias)
            linear1_loss = SquareLossEW(self.linea1_weight[i],model.linear1.weight) + SquareLossEW(self.linear1_bias[i],model.linear1.bias)
            EWSLoss = EWSLoss + conv_1_loss + conv2_loss + conv3_loss + linear1_loss
        return EWSLoss
    def StoreEW(self,index,top_k=2):
         temp = []
         for i in range(len(self.conv1_weight)):
            if i == index:
                temp.append(999)
            else:
                conv_1_loss = SquareLossEW(self.conv1_weight[i], self.conv1_weight[index]) + SquareLossEW(self.conv1_bias[i],self.conv1_bias[index])+ SquareLossEW(self.bn1_weight[i],self.bn1_weight[index])+SquareLossEW(self.bn1_bias[i],self.bn1_bias[index])
                conv2_loss = SquareLossEW(self.conv2_weight[i], self.conv2_weight[index]) + SquareLossEW(self.conv2_bias[i],self.conv2_bias[index])+ SquareLossEW(self.bn2_weight[i],self.bn2_weight[index])+SquareLossEW(self.bn2_bias[i],self.bn2_bias[index])
                conv3_loss = SquareLossEW(self.conv3_weight[i], self.conv3_weight[index]) + SquareLossEW(self.conv3_bias[i],self.conv3_bias[index])+ SquareLossEW(self.bn3_weight[i],self.bn3_weight[index])+SquareLossEW(self.bn3_bias[i],self.bn3_bias[index])
                linear1_loss = SquareLossEW(self.linea1_weight[i],self.linea1_weight[index]) + SquareLossEW(self.linear1_bias[i],self.linear1_bias[index])
                temp.append((conv_1_loss + conv2_loss + conv3_loss + linear1_loss).item())
         strorevector = np.array(temp)
         index = np.argpartition(strorevector,top_k)[:top_k]
         return index
    def LoadParameters(self,index):
        parameters_list = []
        parameters_list.append(self.conv1_weight[index])
        parameters_list.append(self.conv1_bias[index])
        parameters_list.append(self.bn1_weight[index])
        parameters_list.append(self.bn1_bias[index])
        parameters_list.append(self.conv2_weight[index])
        parameters_list.append(self.conv2_bias[index])
        parameters_list.append(self.bn2_weight[index])
        parameters_list.append(self.bn2_bias[index])
        parameters_list.append(self.conv3_weight[index])
        parameters_list.append( self.conv3_bias [index])
        parameters_list.append(self.bn3_weight[index])
        parameters_list.append(self.bn3_bias[index])
        parameters_list.append(self.linea1_weight[index])
        parameters_list.append(self.linear1_bias[index])
        return parameters_list
    def getAvgParameter(self,numbers):
        parameters_list = []
        conv1_weight = 0
        conv1_bias = 0
        bn1_weight = 0
        bn1_bias = 0
        conv2_weight = 0
        conv2_bias = 0
        bn2_weight = 0
        bn2_bias = 0
        conv3_weight = 0
        conv3_bias = 0
        bn3_weight = 0
        bn3_bias = 0
        linear1_weight = 0
        linear1_bias = 0
        for i in range(len(self.conv1_weight)):
            conv1_weight  += self.conv1_weight[i]
            conv1_bias += self.conv1_bias[i]
            bn1_weight += self.bn1_weight[i]
            bn1_bias += self.bn1_bias[i]
            conv2_weight +=self.conv2_weight[i] 
            conv2_bias += self.conv2_bias[i] 
            bn2_weight += self.bn2_weight[i] 
            bn2_bias += self.bn2_bias[i] 
            conv3_weight += self.conv3_weight[i] 
            conv3_bias += self.conv3_bias[i] 
            bn3_weight += self.bn3_weight[i] 
            bn3_bias  += self.bn3_bias[i] 
            linear1_weight += self.linea1_weight[i]
            linear1_bias += self.linear1_bias[i]
        conv1_weight = conv1_weight / numbers
        conv1_bias = conv1_bias / numbers
        bn1_weight = bn1_weight / numbers
        bn1_bias = bn1_bias / numbers
        conv2_weight = conv2_weight / numbers
        conv2_bias = conv2_bias / numbers
        bn2_weight = bn2_weight / numbers
        bn2_bias = bn2_bias / numbers
        conv3_weight = conv3_weight / numbers
        conv3_bias = conv3_bias / numbers
        bn3_weight = bn3_weight / numbers
        bn3_bias = bn3_bias / numbers
        linear1_weight = linear1_weight / numbers
        linear1_bias = linear1_bias / numbers
        parameters_list.append(conv1_weight)
        parameters_list.append(conv1_bias)
        parameters_list.append(bn1_weight)
        parameters_list.append(bn1_bias)
        parameters_list.append(conv2_weight)
        parameters_list.append(conv2_bias)
        parameters_list.append(bn2_weight)
        parameters_list.append(bn2_bias)
        parameters_list.append(conv3_weight)
        parameters_list.append( conv3_bias )
        parameters_list.append(bn3_weight)
        parameters_list.append(bn3_bias)
        parameters_list.append(linear1_weight)
        parameters_list.append(linear1_bias)
        return parameters_list
        
