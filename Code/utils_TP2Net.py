#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:55:37 2020

@author: Demain Wang
"""
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

'''  length of history/future data  '''
def hstfutset(HST_LEN=15,FUT_LEN=25):
    return HST_LEN,FUT_LEN


def seedset(seed):
    '''  Fixing the random seed ensures that the experimental results can be reproduced  '''  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    return


def loaddataset(data_file):
    ''' Read data from the file and do simple preprocessing '''
    assert isinstance(data_file,list)
    X_data=[]
    Y_label=[]

    HST_LEN,FUT_LEN=hstfutset()
    
    # One-hot encoding for driving maneuver
    enc=OneHotEncoder()
    enc.fit(np.array([[0,1,2]]).T)
    for dfid in range(len(data_file)):
        dataraw=pd.read_csv(data_file[dfid])
        dataraw=np.asarray(dataraw)
        
        # get the vehicle ID
        vId=np.unique(dataraw[:,0]).astype(int)
        
        X_temp=[]
        Y_temp=[]
        
        for idx in vId:
        # for debug
#            if idx>50:
#                break

        # get raw data of ID%idx
            data=dataraw[dataraw[:,0]==idx,:]
            
            #if len(data) less than HST_LEN+FUT_LEN-1, it can't compose at least one prece of data
            if HST_LEN+FUT_LEN-1<data.shape[0]:
                #nums of data
                for i in range(data.shape[0]-HST_LEN-FUT_LEN+1):
                    temp=data[i:i+HST_LEN,0:data.shape[1]-2].copy()
                    
                    # Get the origin of the current coordinate system
                    zero_loc=data[i+HST_LEN-1,2:4].copy()
                    
                    # Get the input, lat maneuver and lon maneuver 
                    tempy=data[i+HST_LEN:i+HST_LEN+FUT_LEN,2:4].copy()
                    tempylat=data[i+HST_LEN-1:i+HST_LEN+FUT_LEN-1,data.shape[1]-2].copy()
                    tempylon=data[i+HST_LEN-1:i+HST_LEN+FUT_LEN-1,data.shape[1]-1].copy()
                    
                    #one-hot encode
                    tempylat=enc.transform(np.array([tempylat]).T).toarray()
                    tempylon=enc.transform(np.array([tempylon]).T).toarray()
                    
                    #Unify the coordinate systems
                    for j in range(9):
                        for k in range(HST_LEN):
                            if(temp[k,j*8]!=0):
                                temp[k,(j*8+2):(j*8+4)]-=zero_loc
                    for k in range(FUT_LEN):
                        tempy[k,:]-=zero_loc
                    
                    # delete the vehicle ID                                
                    temp=np.delete(temp,[i*8 for i in range(9)],axis=1)
                    
                    # get the input
                    X_temp.append(temp)
                    Y_temp.append(np.hstack((tempy,tempylat,tempylon)))
        
        # reshape and concatenate
        X_temp=np.reshape(X_temp,(len(X_temp),HST_LEN,-1))
        Y_temp=np.reshape(Y_temp,(len(Y_temp),FUT_LEN,-1))
        if len(X_data)==0:
            X_data=X_temp
            Y_label=Y_temp
        else:
            X_data=np.concatenate((X_data,X_temp),axis=0)
            Y_label=np.concatenate((Y_label,Y_temp),axis=0)
    
    # numpy to torch.tensor
    X_data=torch.tensor(X_data,dtype=torch.float32)
    Y_label=torch.tensor(Y_label,dtype=torch.float32)
    
    return X_data,Y_label



'''  MSEloss for training  '''
class MSEloss_mod(nn.Module):
    def __init__(self):
        super(MSEloss_mod, self).__init__()
        
    def forward(self,y_pred, y_gt):
        muX = y_pred[:,:,0]
        muY = y_pred[:,:,1]
        x = y_gt[:,:, 0].permute(1,0)
        y = y_gt[:,:, 1].permute(1,0)
        out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
        
        #mean loss
        lossVal = torch.sum(out)/(out.shape[0]*out.shape[1])
        
        return lossVal

''' Focal loss for training '''
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, size_average = False):
        super(FocalLoss, self).__init__()
        
        #basic setting
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, lon_pred, lon_gt, lat_pred, lat_gt):
        
        #lon maneuver
        pt_lon = (lon_gt * lon_pred).sum(1) + self.elipson
        logpt_lon=pt_lon.log()
        sub_pt_lon = 1 - pt_lon
        fl_lon = -self.alpha * (sub_pt_lon)**self.gamma * logpt_lon
        
        #lat maneuver
        pt_lat = (lat_gt * lat_pred).sum(1) + self.elipson
        logpt_lat=pt_lat.log()
        sub_pt_lat = 1 - pt_lat
        fl_lat = -self.alpha * (sub_pt_lat)**self.gamma * logpt_lat
        
        #get the total loss
        fl=(fl_lon+fl_lat)
        
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()/fl.size(0)


'''  get the accuracy of maneuver classifiction   '''
def Maneuver_acc(man_pred,man_gt):
    cnt=0
    for i in range(man_pred.size(0)):
        if torch.max(man_gt[i,:],0)[1]==torch.max(man_pred[i,:],0)[1]:
            cnt+=1
            
    return cnt/man_pred.size(0)

















