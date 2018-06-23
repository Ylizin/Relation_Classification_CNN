# coding=utf-8

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, args):
    super(CNN, self).__init__()
    self.dp = args.dp # Position embedding dimension
    self.vac_len_pos = args.vac_len_pos
    self.dw = 300
    self.vac_len_word = args.vac_len_word
    self.vac_len_rel = args.vac_len_rel
    self.dc = args.dc
    self.seq_len = args.seq_len
    self.dropout_rate = args.dropout_rate

    #the first param is the length of the whole wordSet,and the dw is the dimension of each vector,the embedding matrix is vacLen*dw
    #self.word_embedding = nn.Embedding(self.vac_len_word, self.dw) #word embedding input by seq(batch of sentence),and output turn each element into a vector of dw
    
    #self.word_embedding.weight.data.copy_(torch.from_numpy(args.word_embedding))
    self.pos_embedding_pos1 = nn.Embedding(self.vac_len_pos, self.dp)
    self.pos_embedding_pos2 = nn.Embedding(self.vac_len_pos, self.dp)

    self.dropout = nn.Dropout(args.dropout_rate)

    self.convs = nn.ModuleList([nn.Sequential(#dc would be considered as the number of the conv_kernels of each size
        nn.Conv1d(3*self.dw + 2 * self.dp, self.dc, kernel_size=kernel_size,padding=int((kernel_size - 0) / 2)),
        nn.Tanh(),
        nn.MaxPool1d(self.seq_len)
    ) for kernel_size in args.kernel_sizes])#for each kernel size build a conv-tanh-pool operation

    self.fc = nn.Linear(self.dc * len(args.kernel_sizes), self.vac_len_rel)#the num of relations
  
  
  def forward(self, W, W_pos1, W_pos2):#every layer is operated on each data in the batch
   
    #e1 = self.word_embedding(e1)
    #e2 = self.word_embedding(e2)
    #W = self.word_embedding(W)
    W_pos1 = self.pos_embedding_pos1(W_pos1)
    W_pos2 = self.pos_embedding_pos2(W_pos2)
    #if we wanna select 3 word as the feature then we can turn seq into 3dimension 
    #like 128*120*3 and cat the positionFeature at dim=2

    Wa = torch.cat([W, W_pos1, W_pos2], dim=2)

    #before the permute
    #d0 :seq index in batch
    #d1 :word index in seq
    #d2 :vec of the word after permute, it was delivered into conv imply the inChannel param  
    each_conv = [conv(Wa.permute(0, 2, 1)) for conv in self.convs]

    conv = torch.cat(each_conv, dim=1)#cat the output of convs

    conv = self.dropout(conv)

    #e_concat = torch.cat([e1, e2], dim=1)

    all_concat = conv.view(conv.size(0), -1) #view conv res as N(batchsize),*(any dimension)

    out = self.fc(all_concat)
    #print("out before softmax:{0}".format(out.size()))
    out = F.softmax(out,dim = 1) #softmax at column direction 
    #print("out after softmax:{0}".format(out.size()))
    #print("sum of the after out : {0}".format(torch.sum(out,dim = 1)))
    return out

