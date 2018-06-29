# coding=utf-8
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset

import gensim
from gensim.models import KeyedVectors

model = None

dw = 300

def loadWordVector(Path):
    global model
    print('Word Vec loading:\n')
    model = KeyedVectors.load_word2vec_format(Path,binary = True)
    print('Word load complete.\n')

def word2Vec(Word):
    try:
        return model[Word]
    except KeyError:
        return np.zeros(dw)


def concatSeq(i,seq,minLength,maxLength):
    seq_i = None
    if i == minLength :
        seq_i = np.zeros(dw)
        seq_i = np.append(seq_i,seq[i])
        seq_i = np.append(seq_i,seq[i+1])
    elif i < maxLength-1 and i> minLength:
        seq_i = seq[i-1]
        seq_i = np.append(seq_i,seq[i])
        seq_i = np.append(seq_i,seq[i+1])
    elif i == maxLength-1 : 
        seq_i = seq[i-1]
        seq_i = np.append(seq_i,seq[i])
        seq_i = np.append(seq_i,np.zeros(dw))
    else:
        raise IndexError("concatSeq index incorrect!")
    return seq_i


class SemEvalDataset(Dataset):
  def __init__(self, filename, max_len, d=None):#d for dictionary 
    loadWordVector("./data/vec.bin")
    seqs, e1_pos, e2_pos, rs = load_data(filename)
    self.max_len = max_len
    #if d is None:
    #  self.d = build_dict(seqs)
    if d is None :
        self.rel_d = build_dict([[r] for r in rs], add_extra=False)
    else :
        self.rel_d = d
        #else:
    #  self.d = d[0]
    #  self.rel_d = d[1]
    self.seqs, self.e1s, self.e2s, self.dist1s, self.dist2s =\
      self.vectorize_seq(seqs, e1_pos, e2_pos)
    self.rs = np.array([[self.rel_d.word2id[r]] for r in rs]) #the relation dict can turn relations into a index,for the following operations
  
  def vectorize_seq(self, seqs, e1_pos, e2_pos):
    new_seqs = np.zeros((len(seqs), self.max_len,3*dw))
    dist1s = np.zeros((len(seqs), self.max_len))
    dist2s = np.zeros((len(seqs), self.max_len))
    e1s = np.zeros((len(seqs), dw))
    e2s = np.zeros((len(seqs), dw))
    for r, (seq, e1_p, e2_p) in enumerate(zip(seqs, e1_pos, e2_pos)):
      seq = list(map(word2Vec, seq)) #convert each element of seq into a tensor/vector
      dist1 = list(map(map_pos, [idx - e1_p[1] for idx, _ in enumerate(seq)])) 
      dist2 = list(map(map_pos, [idx - e2_p[1] for idx, _ in enumerate(seq)]))
      e1s[r] = seq[e1_p[1]]
      e2s[r] = seq[e2_p[1]]
      
      for i in range(min(len(seq),self.max_len)):
        new_seqs[r,i] = concatSeq(i,seq,0,min(len(seq),self.max_len))
        dist1s[r,i] = dist1[i]
        dist2s[r,i] = dist2[i]
      #do piece wise here, we divide a sentence into three part and treat them as different sentences but they are neighbors
      #if(e1_p != 0 ):#if(e1_p == 0) the first part is none 
      #  for i in range(0,e1_p):
      #    new_seqs[3*r, i] = concatSeq(i,seq,0,e1_p) #new_seqs is a 3D matrix of sentenceNum,sentenceLength,wordVecDim (= 3)
      #    dist1s[3*r, i] = dist1[i]
      #    dist2s[3*r, i] = dist2[i]
      #if(e1_p != e2_p):#if(e1_p==e2_p) the second part is none
      #    for i in range(e1_p,e2_p):
      #      new_seqs[3*r+1, i] = concatSeq(i,seq,e1_p,e2_p) 
      #      dist1s[3*r+1, i] = dist1[i]
      #      dist2s[3*r+1, i] = dist2[i]
      #if(e2_p <= min(len(seq),self.max_len)):
      #    for i in range(e2_p,min(len(seq),self.max_len)):
      #      new_seqs[3*r+2, i] = concatSeq(i,seq,e2_p,min(len(seq),self.max_len))
      #      dist1s[3*r+2, i] = dist1[i]
      #      dist2s[3*r+2, i] = dist2[i]
    return new_seqs, e1s, e2s, dist1s, dist2s #new seqs = 3*seqs

  def __len__(self):
    return len(self.seqs)
  
  def __getitem__(self, index):
    seq = torch.from_numpy(self.seqs[index]).float()#for each iteration , return a seq of a sentence,
                                                #and batch will do 128(--bz) iteration to get
                                                #128 sentences through 'index'
    e1 = torch.from_numpy(self.e1s[index]).float() #the embedding layers require longTensor as index
    e2 = torch.from_numpy(self.e2s[index]).float()
    dist1 = torch.from_numpy(self.dist1s[index]).long()
    dist2 = torch.from_numpy(self.dist2s[index]).long()
    r = torch.from_numpy(self.rs[index]).long()
    return seq, e1, e2, dist1, dist2, r

'''
    load the data processed by the preprocess
'''
def load_data(filename):
  seqs = []
  e1_pos = []
  e2_pos = []
  rs = []
  with open(filename, 'r') as f:
    for line in f:
      data = line.strip().split('\t')
      while " " in data[0]:
        data[0] = data[0].lower().split(' ')
      seqs.append(data[0])
      e1_pos.append((int(data[1]), int(data[2])))
      e2_pos.append((int(data[3]), int(data[4])))
      rs.append(data[5])
  return seqs, e1_pos, e2_pos, rs

class Dictionary(object):
  def __init__(self):
    self.word2id = {} #dict
    self.id2word = [] #list
  def add_word(self, word):
    if word not in self.word2id:
      self.word2id[word] = len(self.id2word) #append at the end of id2word and word2id,with the position of len(*)
      self.id2word.append(word)


'''
    build the id2word and word2id by the whole seqs
    we need it here to build the index for relations
'''
def build_dict(seqs, add_extra=True, dict_size=100000):
  d = Dictionary()
  cnt = Counter()
  for seq in seqs:
    cnt.update(seq)
  d = Dictionary()
  if add_extra:
    d.add_word(None) # 0 for not in the dictionary
  for word, cnt in cnt.most_common()[:dict_size]:
    d.add_word(word)
  return d

'''
    map the pos to entities to the input index to the embedding layers
'''
def map_pos(p):
  if p < -60:
    return 0
  elif -60 <= p < 60:
    return p + 61
  else:
    return 121

'''
    cause we do not need to build a word2Vec here 
    we simply use the output of the google
'''
#def load_embedding(embedding_file, word_list_file, d):
#  word_list = {}
#  with open(word_list_file) as f:
#    for i, line in enumerate(f):
#      word_list[line.strip()] = i

#  with open(embedding_file, 'r') as f:
#    lines = f.readlines()
#  def process_line(line):
#    return list(map(float, line.split(' ')))
#  lines = list(map(process_line, lines))

#  val_len = len(d.id2word)
#  dw = len(lines[0])
#  embedding = np.random.uniform(-0.01, 0.01, size=[val_len, dw])
#  num_pretrained = 0
#  for i in range(1, val_len):
#    if d.id2word[i] in word_list:
#      embedding[i,:] = lines[word_list[d.id2word[i]]]
#      num_pretrained += 1
  
#  print('#pretrained: {}, #vocabulary: {}'.format(num_pretrained, val_len))
#  return embedding
