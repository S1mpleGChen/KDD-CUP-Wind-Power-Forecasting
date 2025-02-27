#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:02:02 2022

@author: chenguibin
"""
import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.io import Dataset
from paddle.static import InputSpec

import paddle.fluid as fluid

from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import Layer, to_variable
import paddle.fluid.dygraph as dygraph
import gc
import numpy as np




class BaselineGruModel(nn.Layer):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGruModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=True)   #time major为True时，输入lstm的形式就是seq,batch,dims
        self.projection = nn.Linear(self.hidR, self.out)

    def forward(self, x_enc):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:  就是batch_x
        Returns:
            A tensor
        """
        x = paddle.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])   #初始化hidden_state,因为第一个LSTM层没有能接收的隐状态
        x_enc = paddle.concat((x_enc, x), 1)
        x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))
        dec, _ = self.lstm(x_enc)
        dec = paddle.transpose(dec, perm=(1, 0, 2))
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out:]
        return sample  # [B, L, D]  batch, output_length, features_dimsion


class GRUEncoder(paddle.nn.Layer):
    def __init__(self, settings):
        super(GRUEncoder, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=True)   #time major为True时，输入lstm的形式就是seq,batch,dims
        

    def forward(self, x_enc):
        x = paddle.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])   #初始化hidden_state,因为第一个LSTM层没有能接收的隐状态
        x_enc = paddle.concat((x_enc, x), 1)
        x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))  #seq. batch, dims
        enc_lstmout, _ = self.lstm(x_enc)
        return enc_lstmout


#使用注意力机制的GRU序列预测模型    
class Atten_En_Decoder(paddle.nn.Layer):
    def __init__(self, settings,use_teacher_forcing = True):
        super(Atten_En_Decoder, self).__init__()
        
        #self.dims_no_label = settings["future_dims"]
        self.teacher_forcing_ratio = settings["teacher_ratio"]  #以上这两个数量需要改代码设置
        self.output_len = settings["output_len"]
        self.input_len = settings["input_len"]
        self.use_teacher_forcing = use_teacher_forcing
        self.out = settings["out_var"]
        self.hidC = settings["in_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        
        self.hidR = 48
        #encoder
        self.lstm1 = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"])   #time major为True时，输入lstm的形式就是seq,batch,dims
        #decoder
        self.lstm2 = nn.GRU(input_size= 3 + self.hidR,  #接收未来特征和encoder的输出
                                   hidden_size=self.hidR, num_layers=1)
        
        #MLP
        self.projection = nn.Linear(self.hidR, self.out)
        
        

        # for computing attention weights
        self.attention_linear1 = paddle.nn.Linear(self.hidR * 2, self.hidR)  #接收encoder输出和decoder时间序列点中的上一个hidden state
        self.attention_linear2 = paddle.nn.Linear(self.hidR, 1)
        
        # for computing output logits
        self.outlinear =paddle.nn.Linear(self.hidR, self.output_len)
        
    def forward(self, x_enc, xf, target):   #target是这个序列的真实预测值：[batch,outlen,1],1是预测的label维度
        """
        xf:future features
        previous hidden:
        """
        
        batch_size = xf.shape[0]
        ##encoder output
        #initilize encoder hidden input
        #x = paddle.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])   #初始化hidden_state,因为第一个LSTM层没有能接收的隐状态
        #x_enc = paddle.concat((x_enc, x), 1)
        #x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))  #seq. batch, dims
        enc_lstmout, _ = self.lstm1(x_enc)  #seq, batch, dims
        
        ##decoder
        # Creating first decoder_hidden_state = 0
        decoder_hidden = paddle.zeros(shape=[batch_size, 1, self.hidR], dtype="float32")  #逐条输入，因此num_seq=1
        # Initializing predictions vector
        outputs = paddle.zeros(shape=[batch_size, self.output_len, 1], dtype="float32")

        # Initializing first prediction
        decoder_output = paddle.zeros(shape=[batch_size, 1, 1],dtype="float32")  #逐条输入，因此num_seq=1

        # List of alphas, for attention check
        attn_list = []
        #print('num layers of lstm2:', self.lstm2.num_layers)
        #print('hidden state size:',decoder_hidden.shape)

        for t in range(self.output_len):
            #enc_lstmout = paddle.transpose(enc_lstmout, [1,0,2])
            #decoder_hidden = paddle.transpose(decoder_hidden, [1,0,2])
            attention_inputs = paddle.concat((enc_lstmout, 
                                      paddle.tile(decoder_hidden, repeat_times=[1, self.input_len, 1])),
                                      axis=-1
                                     )
            attention_hidden = self.attention_linear1(attention_inputs)
            attention_hidden = F.tanh(attention_hidden)
            attention_logits = self.attention_linear2(attention_hidden)
            attention_logits = paddle.squeeze(attention_logits)
            
            attention_weights = F.softmax(attention_logits) 
            attn_list.append(attention_weights)
            attention_weights = paddle.expand_as(paddle.unsqueeze(attention_weights, -1), 
                                             enc_lstmout)
            
            context_vector = paddle.multiply(enc_lstmout, attention_weights)               
            context_vector = paddle.sum(context_vector, 1)
            context_vector = paddle.unsqueeze(context_vector, 1)
        
            x_input = paddle.concat((xf[:,t,:].unsqueeze(1), decoder_output,context_vector), axis=-1)   #这里可能xf需要经过mlp变换一下输出的num_dims
            
            # GRU decoder
            #GRU input previous hidden state requirements: num_layers*num_directions,batch, dims
            decoder_hidden = paddle.transpose(decoder_hidden, [1,0,2])
            #print('num layers of lstm2:', self.lstm2.num_layers)
            #print('hidden state size:',decoder_hidden.shape)
            dec_lstmout, d_hidden = self.lstm2(x_input, decoder_hidden)  #
            #decoder_out = paddle.transpose(dec_lstmout, perm=(1, 0, 2))
            decoder_output = self.projection(self.dropout(dec_lstmout))
            decoder_hidden = paddle.transpose(d_hidden, [1,0,2])
            #decoder_hidden = d_hidden  #再翻转回来给下一个循环步使用

            outputs[:,t,:] = decoder_output.squeeze(1)
             # Teacher forcing
            teach_forcing = True if random.random() < self.teacher_forcing_ratio else False
            #print('Use teacher!:', self.teacher_forcing_ratio)
            #print('teach forcing!:', teach_forcing)

            if self.use_teacher_forcing and teach_forcing:
                #print('use_teacher_forcing:!!')
                
                decoder_output = target[:,t,:].unsqueeze(1)

       

        return outputs, attn_list
            


