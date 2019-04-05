import tensorflow as tf
from tensorflow.python.lib.io import file_io

import numpy as np
import os

def vocabulary(texts, filter=[]):
    vocab_dict = {}
    ix = 0
    for sentence in texts:
        for wrd in sentence:
            if wrd not in vocab_dict.keys() and wrd not in filter:
                vocab_dict[wrd] = ix
                ix += 1
    return vocab_dict

def BoW_embedding(texts, vocab, normalize=True):
    ohe = []
    for txt in texts:
        v = np.zeros(len(vocab))
        for l in txt:
            if l not in vocab.keys():
                continue
            ix = vocab[l]
            v[ix] += 1
        if normalize:
            if sum(v) == 0:
                print(txt)
            v = v/sum(v)
        ohe.append(v)
    return np.array(ohe)

def num_encode(texts, vocab):
    ne = []
    for txt in texts:
        v = np.zeros(len(txt))
        for ix, x in enumerate(txt):
            v[ix] = vocab[x]
        ne.append(v)
    return ne
def read_data(filename):
    
  with file_io.FileIO(filename, mode='r') as f:
    tmp = f.read()
  tmp = tmp.split('\n')
  data = []
  for row in tmp[:-1]:
    A, B, E, _, _, _ = tuple(row.split(','))
    data.append([A, B, int(E)])
  return np.array(data)

def batch_data(data, batch_size):
  np.random.shuffle(data)
  num_batches = int(len(data)//batch_size)-1
  data = data.T
  for i in range(num_batches):
    A = data[0][i*batch_size:(i+1)*batch_size]
    B = data[1][i*batch_size:(i+1)*batch_size]
    E = data[2][i*batch_size:(i+1)*batch_size]
    yield list(A), list(B), list(E.astype(np.float32))
    
def grab_data(batch_size):
    # fetch a generator
    filename = r'..\Logical_Entailment\temp\train.txt'
    return batch_data(read_data(filename), batch_size)