
# coding=utf-8

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
import argparse
from CNN import CNN

from LoadData import *
#import evaluate

rel_TP = torch.zeros(19)
rel_FP = torch.zeros(19)
rel_FN = torch.zeros(19)

def zero_rels():
    global rel_TP
    global rel_FP
    global rel_FN
    rel_FN = torch.zeros_like(rel_FN)
    rel_FP = torch.zeros_like(rel_FP)
    rel_TP = torch.zeros_like(rel_TP)

def getPR():
    PR = 0 
    #macro avg F1
    rel_TP[1] = rel_TP[1] + rel_TP[18]
    rel_FN[1] = rel_FN[1] + rel_TP[18]
    rel_FP[1] = rel_FP[1] + rel_FP[18]

    rel_TP[2] = rel_TP[2] + rel_TP[11]
    rel_FN[2] = rel_FN[2] + rel_FN[11]
    rel_FP[2] = rel_FP[2] + rel_FP[11]

    rel_TP[3] = rel_TP[3] + rel_TP[17]
    rel_FN[3] = rel_FN[3] + rel_FN[17]
    rel_FP[3] = rel_FP[3] + rel_FP[17]

    rel_TP[4] = rel_TP[4] + rel_TP[14]
    rel_FN[4] = rel_FN[4] + rel_FN[14]
    rel_FP[4] = rel_FP[4] + rel_FP[14]

    rel_TP[5] = rel_TP[5] + rel_TP[15]
    rel_FN[5] = rel_FN[5] + rel_FN[15]
    rel_FP[5] = rel_FP[5] + rel_FP[15]

    rel_TP[6] = rel_TP[6] + rel_TP[7]
    rel_FN[6] = rel_FN[6] + rel_FN[7]
    rel_FP[6] = rel_FP[6] + rel_FP[7]     

    rel_TP[7] = rel_TP[8] + rel_TP[16]
    rel_FN[7] = rel_FN[8] + rel_FN[16]
    rel_FP[7] = rel_FP[8] + rel_FP[16]

    rel_TP[8] = rel_TP[9] + rel_TP[12]
    rel_FN[8] = rel_FN[9] + rel_FN[12]
    rel_FP[8] = rel_FP[9] + rel_FP[12]

    rel_TP[9] = rel_TP[10] + rel_TP[13]
    rel_FN[9] = rel_FN[10] + rel_FN[13]
    rel_FP[9] = rel_FP[10] + rel_FP[13]

    for i,(tp,fp,fn) in enumerate(zip(rel_TP,rel_FP,rel_FN)):
        f1 = 0 
        if i ==0 or i>9 :#ignore 'other' relation 
          continue
        if tp!= 0:
          p = tp/(tp+fp)
          r = tp/(tp+fn)
          f1 = 2*p*r/(p+r)
        PR += f1
    
    return PR/9

def accuracy(preds, labels):
  total = preds.size(0)
  #preds 1D ,same shape as labels
  preds = torch.max(preds, dim=1)[1] #get the max probability [1] means to get the index of the maxProbable relIndex, see the output of torch.max 
  correct = (preds == labels).sum()
  acc = correct.cpu().item() / total
  for (pred,r) in zip(preds,labels):
      if pred == r:
          rel_TP[pred] = rel_TP[pred]+1
      else:#pred do a wrong classification
          rel_FN[r] = rel_FN[r]+1#should be r,but not be classified as r
          rel_FP[pred] = rel_FP[pred]+1#should not be relation->'pred', but be classified as pred
  return acc

def main():
  parser = argparse.ArgumentParser("CNN")
  parser.add_argument("--dp", type=int, default=30)#word dimension should be scaled as the dw scaled
  parser.add_argument("--dc", type=int, default=192)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--seq_len", type=int, default=120)
  parser.add_argument("--vac_len_rel", type=int, default=19) #the number of relation kind 
  parser.add_argument("--nepoch", type=int ,default=500)
  parser.add_argument("--num_workers", type=int, default=0)#multiprocessing may cause unreasonable errors
  parser.add_argument("--eval_every", type=int, default=10)
  parser.add_argument("--dropout_rate", type=float, default=0.4)
  parser.add_argument("--bz", type=int, default=128)
  parser.add_argument("--kernel_sizes", type=str, default="3,4,5")
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--train_filename', default='./data/train.txt')
  parser.add_argument('--test_filename', default='./data/test.txt')
  parser.add_argument('--model_file', default='./cnn.pt')
  parser.add_argument('--embedding_filename', default='./data/embeddings.txt')
  parser.add_argument('--embedding_wordlist_filename', default='./data/words.lst')

  args = parser.parse_args()
  args.kernel_sizes = list(map(int, args.kernel_sizes.split(',')))
  # Initilization
  torch.cuda.set_device(args.gpu)

  # Load data
  dataset = SemEvalDataset(args.train_filename, max_len=args.seq_len)
  dataloader = DataLoader(dataset, args.bz, True, num_workers=args.num_workers) #128 sentence per batch

  dataset_val = SemEvalDataset(args.test_filename, max_len=args.seq_len, d=dataset.rel_d)
  dataloader_val = DataLoader(dataset_val, args.bz, True, num_workers=args.num_workers)
  #args.word_embedding = load_embedding(args.embedding_filename, args.embedding_wordlist_filename,\
  #  dataset.d)
  args.vac_len_pos = 122
  #args.vac_len_word = len(dataset.d.word2id)
  args.vac_len_rel = len(dataset.rel_d.word2id)
  args.dw = 300
  
  model = CNN(args)
  if(torch.cuda.is_available()):
    model = model.cuda()
  loss_func = nn.CrossEntropyLoss()
  # optimizer = optim.SGD(model.parameters(), lr=0.2)
  optimizer = optim.Adam(model.parameters(), lr = args.lr,weight_decay= 1e-5)
  scheduler = StepLR(optimizer,step_size = 100,gamma = 0.3)
  best_eval_acc = 0.

  for i in range(args.nepoch):
    scheduler.step()
    zero_rels()
    # Training
    total_loss = 0.
    total_acc = 0.
    ntrain_batch = 0
    model.train()
    for (seq, e1, e2, dist1, dist2, r,e1_p,e2_p) in dataloader:
      ntrain_batch += 1
      if(torch.cuda.is_available()):
        seq = seq.cuda()
        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        r = r.cuda() 
        e1 = e1.cuda()
        e2 = e2.cuda()
      r = r.view(r.size(0))#(batchsize)1D
      #r.size(0) means the batch size, and the input of this lossFunction is 
      # input : a tensor with N*C shape(N->batchsize,C ->  the probability of the possible classes)
      # target: N shape tensor, with the index of the class of the target

      #seq.requires_grad_(True)
      #if i ==0 :
        #optimizer.add_param_group({'params':seq,'lr':0.1})#TUNE google pretrained vec
      #print("r : {0}".format(r.size()))
      pred = model(seq, dist1, dist2,e1,e2,e1_p,e2_p)
      #print("pred:{0}".format(pred.size()))
      l = loss_func(pred, r)
      acc = accuracy(pred, r)
      total_acc += acc
      total_loss += l.item()

      optimizer.zero_grad()
      l.backward()
      optimizer.step()
    print("Epoch: {}, Training loss : {:.4}, acc: {:.4} ,PR:{:.4}".\
      format(i, total_loss/ntrain_batch, total_acc / ntrain_batch,getPR()))

    # Evaluation
    if i % args.eval_every == args.eval_every - 1:
      zero_rels()
      val_total_acc = 0.
      nval_batch = 0
      model.eval()
      for (seq, e1, e2, dist1, dist2, r,e1_p,e2_p) in dataloader_val:
        nval_batch += 1
        if(torch.cuda.is_available()):
            seq = seq.cuda()
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            r = r.cuda()
            e1 = e1.cuda()
            e2 = e2.cuda()

        r = r.view(r.size(0))

        pred = model(seq, dist1, dist2,e1,e2,e1_p,e2_p)
        acc = accuracy(pred, r)
        val_total_acc += acc
      best_eval_acc = max(best_eval_acc, val_total_acc/nval_batch)
      print("Epoch: {}, Val acc: {:.4f}, PR:{:.4}".\
        format(i, val_total_acc/nval_batch,getPR()))
  print(best_eval_acc)
  torch.save(model.state_dict(), args.model_file)

if __name__ == '__main__':
  main()