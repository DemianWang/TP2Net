#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Aug  6 18:53:28 2020

@author: Demain Wang
"""
import torch
import torch.nn as nn
from utils_TP2Net import hstfutset

'''  
TP2Net for trajectory prediction 
'''
class TP2Net(nn.Module):
    def __init__(self):
        super(TP2Net,self).__init__()
        
        # basic setting
        self.HST_LEN,self.FUT_LEN=hstfutset()
        self.input_size=7
        self.hidden_size=128
        self.input_embedding_size=64
        self.batch_size=128
        self.ego_enc_size=64
        self.ego_sur_size=64
        self.num_lat_class=3
        self.num_lon_class=3
        self.atten_output_size_tpa=64
                
        # base function
        self.ip_emb_ego=torch.nn.Linear(self.input_size,self.input_embedding_size)
        self.ip_emb_sur_1=torch.nn.Linear(self.input_size,self.input_embedding_size)
        self.in_lstm_ego=torch.nn.LSTM(self.input_embedding_size,self.hidden_size,1,batch_first=True)
        self.in_lstm_sur=torch.nn.LSTM(self.input_embedding_size,self.hidden_size,1,batch_first=True)
        self.ot_emb_ego=torch.nn.Linear(self.hidden_size,self.ego_enc_size)
        self.ot_emb_sur=torch.nn.Linear(self.hidden_size,self.ego_sur_size)
          
        self.dec_lstm = torch.nn.LSTM(self.ego_enc_size + self.ego_sur_size*4 + self.atten_output_size_tpa, self.hidden_size,1)
        self.op_lat = torch.nn.Linear(self.ego_enc_size + self.ego_sur_size*4 + self.atten_output_size_tpa, self.num_lat_class)
        self.op_lon = torch.nn.Linear(self.ego_enc_size + self.ego_sur_size*4 + self.atten_output_size_tpa, self.num_lon_class)
        self.op=torch.nn.Linear(self.hidden_size,5)
        
        # TPA
        self.TPA=TemporalPatternAttention(self.input_size,self.batch_size,self.atten_output_size_tpa)
        
        # ego_zero
        self.egoempty_enc=torch.zeros((self.batch_size,self.ego_sur_size)).cuda()    
        
        # VOI weighting
        self.conv3x3=torch.nn.Conv2d(self.ego_sur_size, self.ego_sur_size*4, (3,3))
        self.conv1x1=torch.nn.Conv2d(self.ego_sur_size, self.ego_sur_size, (1,1))
        
        # activation module
        self.leaky_relu=torch.nn.LeakyReLU(0.1)
        self.relu=torch.nn.ReLU()
        self.softmax=torch.nn.Softmax(dim=1)

    def forward(self,ego,pre,fol,lftpre,lftalo,lftfol,rgtpre,rgtalo,rgtfol):
        
        #encoding of ego vehicle
        hst_ego=self.ip_emb_ego(ego)     
        _,(hid_ego,_)=self.in_lstm_ego(self.leaky_relu(hst_ego))
        ego_enc=self.leaky_relu(self.ot_emb_ego(torch.squeeze(hid_ego)))
        
        #encoding of sur vehicle
        pre_enc,hst_pre=self.sudveh_enc(pre)
        fol_enc,hst_fol=self.sudveh_enc(fol)
        lftpre_enc,hst_lftpre=self.sudveh_enc(lftpre)
        lftalo_enc,hst_lftalo=self.sudveh_enc(lftalo)
        lftfol_enc,hst_lftfol=self.sudveh_enc(lftfol)
        rgtpre_enc,hst_rgtpre=self.sudveh_enc(rgtpre)
        rgtalo_enc,hst_rgtalo=self.sudveh_enc(rgtalo)
        rgtfol_enc,hst_rgtfol=self.sudveh_enc(rgtfol) 

        # VOI weighting
        pre=torch.stack((lftpre_enc,pre_enc,rgtpre_enc),2).contiguous()
        alo=torch.stack((lftalo_enc,self.egoempty_enc,rgtalo_enc),2).contiguous()
        fol=torch.stack((lftfol_enc,fol_enc,rgtfol_enc),2).contiguous()
        sur_cs=torch.stack((pre,alo,fol),2)
        
        # VOI conv 
        sur_cs=self.leaky_relu(self.conv1x1(sur_cs))        
        sur_conv3x3=self.leaky_relu(self.conv3x3(sur_cs))
        sur_conv3x3=torch.squeeze(sur_conv3x3)
        
        # Temporal pattern attention
        tpa_attention=self.TPA(ego)
        
        # concatenate
        enc=torch.cat((ego_enc,tpa_attention,sur_conv3x3),1)

        # maneuver recognition
        lat_pred = self.softmax(self.op_lat(enc))
        lon_pred = self.softmax(self.op_lon(enc))
        
        # predict the trajectory
        fut_pd=self.decode(enc)
        
        return fut_pd,lat_pred,lon_pred
    
    def sudveh_enc(self,veh):
        
        #encoding of sur vehicle
        sur_lin=self.ip_emb_sur_1(veh)
        _,(hid_veh,_)=self.in_lstm_sur(self.leaky_relu(sur_lin))
        veh_enc=self.leaky_relu(self.ot_emb_sur(torch.squeeze(hid_veh)))
        
        return veh_enc,sur_lin

    def decode(self,enc):
        
        #decode the tensor
        enc=enc.repeat(self.FUT_LEN,1,1)
        h_dec,_=self.dec_lstm(enc)
        h_dec=h_dec.permute(1,0,2)
        fut_pred=self.op(h_dec)
        fut_pred=fut_pred.permute(1,0,2)
        fut_pred=outputActivation(fut_pred)
        
        return fut_pred

# It can also be used for NLL loss
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    
    return out

# Temporal Pattern attintion for trajectory prediction 
# Adaptation changes have been made 

class TemporalPatternAttention(nn.Module):
    def __init__(self,input_size_atten,batch_size,output_size):
        super(TemporalPatternAttention,self).__init__()
        ''' 
        input: 
            input_size_atten: the size of attention input
            batch_size: the size of mini_batch
            output_size: size of ht'
        '''
        #basic setting
        self.input_size_attention=input_size_atten
        self.atten_hid_size=output_size
        self.window_length,self.FUT_LEN=hstfutset()
        self.output_channel=32
        self.drop_rate=0.3
        self.batch_size=batch_size
        
        #basic function
        self.in_lstm_attention=torch.nn.LSTM(self.input_size_attention,self.atten_hid_size,1,batch_first=True)
        self.compute_convolution=torch.nn.Conv2d(1,self.output_channel,kernel_size=(1,self.atten_hid_size))
        
        # weighting to be register
        self.attention_matrix = nn.Parameter(torch.ones(self.batch_size, self.output_channel, self.atten_hid_size, requires_grad=True))
        self.register_parameter('atten_mat',self.attention_matrix)
        self.final_state_matrix = nn.Parameter(torch.ones(self.batch_size, self.atten_hid_size, self.atten_hid_size, requires_grad=True))
        self.register_parameter('final_state_mat',self.final_state_matrix)
        self.context_vector_matrix = nn.Parameter(torch.ones(self.batch_size, self.atten_hid_size, self.output_channel, requires_grad=True))
        self.register_parameter('context_vector_mat',self.context_vector_matrix)
        
        # weighting init
        torch.nn.init.xavier_uniform_(self.attention_matrix)
        torch.nn.init.xavier_uniform_(self.final_state_matrix)
        torch.nn.init.xavier_uniform_(self.context_vector_matrix)
        
        #activation module
        self.relu=torch.nn.ReLU()
        self.dropout=torch.nn.Dropout(self.drop_rate)
        self.leaky_relu=torch.nn.LeakyReLU(0.1)
        
    def forward(self,ego):
        # get the attention input and encode
        atten_input=ego
        # atten_input=self.leaky_relu(ego)
        lstm_hidden,(h_all,_)=self.in_lstm_attention(atten_input)
        
        # reshape
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))
                
        # conv of tensor of op
        output_realigned = lstm_hidden.contiguous()
        hn = hn.permute(1, 0, 2).contiguous()
        input_to_convolution_layer = output_realigned.view(-1, 1, self.window_length, self.atten_hid_size)
        convolution_output = self.leaky_relu(self.compute_convolution(input_to_convolution_layer))
        convolution_output = self.dropout(convolution_output)
                
        #gen attenion map
        convolution_output = convolution_output.squeeze(3)              
        final_hn = torch.zeros(self.attention_matrix.size(0), 1, self.atten_hid_size)
        input_sur = torch.zeros(self.attention_matrix.size(0), atten_input.size(1), atten_input.size(2))
        final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.output_channel, self.window_length)
        diff = 0
        if (hn.size(0) < self.attention_matrix.size(0)):
            final_hn[:hn.size(0), :, :] = hn
            final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
            input_sur[:atten_input.size(0), :, :] = atten_input
            diff = self.attention_matrix.size(0) - hn.size(0)
        else:
            final_hn = hn
            final_convolution_output = convolution_output
            input_sur = atten_input
        
        # key queries and get the value
        convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous().cuda()
        final_hn_realigned = final_hn.permute(0, 2, 1).contiguous().cuda()
        mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix).cuda()
        
        #get the scoring function
        scoring_function = torch.bmm(mat1, final_hn_realigned).cuda()
        alpha = torch.sigmoid(scoring_function)
        
        #broadcast
        context_vector = alpha * convolution_output_for_scoring
        context_vector = torch.sum(context_vector, dim=1).cuda()
                
        # get ht'
        context_vector = context_vector.view(-1, self.output_channel, 1)
        h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix, context_vector)
        
        #reshape        
        h_intermediate=h_intermediate.squeeze(2)#128,32
        
        return h_intermediate









