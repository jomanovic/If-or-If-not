# -*- coding: utf-8 -*-
"""
Baseline models: Linear and Multi-Layer Percepton
"""

import tensorflow as tf
import numpy as np
import os
import data 

from utils import vocabulary, BoW_embedding, num_encode

def MLP(hidden_dims):
    model = tf.keras.Sequential()        
    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def BoW_data(datadir='../Logical_Entailment/temp', filename='train.txt'):
    A, B, y = zip(*data.read_data(os.path.join(datadir, filename)))
    X_1 = BoW_embedding(A, vocabulary(A))
    X_2 = BoW_embedding(B, vocabulary(A))
    X = np.concatenate((X_1, X_2), axis=1)
    y = np.array(y, dtype=np.int32)
    return X, y 

if __name__ == '__main__':
    
    model = MLP([64,64])
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['binary_accuracy'])

    X, y = BoW_data(filename='train.txt')
    X_val, y_val = BoW_data(filename='validate.txt')

    model.fit(X,
              y,
              epochs=20,
              batch_size=32,
              validation_data=(X_val, y_val),
              verbose=2)
    
