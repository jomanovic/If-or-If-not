# -*- coding: utf-8 -*-
"""
We implement efficent relational LSTM and BidirLSTM models for benchmarking.
Hyperparameter optimization is performed with,
Hyperopt + Keras : https://github.com/maxpumperla/hyperas
"""

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from tensorflow.keras.layers import Input, Masking, Dense, Embedding, Bidirectional, LSTM
from tensorflow.keras.layers import Flatten, Reshape, concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import TensorBoard

import tensorflow as tf
import numpy as np

from utils import vocabulary, to_numeric

def data():

    A_0, B_0, y_0 = zip(*data.read_data('../Logical_Entailment/temp/train.txt'))
    A_1, B_1, y_1 = zip(*data.read_data('../Logical_Entailment/temp/validate.txt'))
    y_0, y_1 = np.array(y_0), np.array(y_1)

    X = [a+'_'+b for a, b in zip(*[A_0,B_0])]
    X_val = [a+'_'+b for a, b in zip(*[A_1,B_1])]
    
    vocab = vocabulary(X)
    
    X = to_numeric(X, vocab)
    X_val = to_numeric(X_val, vocab)
    
    maxlen = len(max(*X, key=len))
    vocab_size = len(vocab)
    
    #print('Pad sequences A, B')

    X = pad_sequences(X, padding='post', maxlen=maxlen)
    y = np.array(y_0, dtype=int)

    
    X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
    y_val = np.array(y_1, dtype=int)
    
    return X, y, X_val, y_val, maxlen, vocab_size

def LSTM(X, y, X_val, y_val, maxlen, vocab_size):
    
    with tf.device('/GPU:0'):      
        lr = {{choice([1e-3, 1e-4, 1e-5])}} # Learning rate
        epochs = {{choice([10])}} # Number of epochs
        batch_size = {{choice([32, 64])}} # Batch size 
        lstm_dim = {{choice([32, 64])}} # LSTM dimension
        emb_dim = {{choice([32, 64])}} # Embedding dimension
        num_dnn = {{choice([1, 2])}} # Size of outer DNN layers
        dense_dim = {{choice([32, 64])}}
    
        
        input = Input(shape=(maxlen,))
        x = Embedding(input_dim=vocab_size,
                      output_dim=emb_dim,
                      input_length=maxlen)(input)  
        x = LSTM(lstm_dim, return_sequences = False)(x)
        for i in range(num_dnn):
            x = Dense(dense_dim, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=[input], outputs=[output])
        #tensorboard = TensorBoard(log_dir=param)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        
        result = model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              #callbacks=[tensorboard],
              verbose=2,
              validation_data = (X_val, y_val))
            
        validation_acc = np.amax(result.history['val_acc']) 
    
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
    
def biLSTM(X, y, X_val, y_val, maxlen, vocab_size):
    
    with tf.device('/GPU:0'):      
        lr = {{choice([1e-4])}} # Learning rate
        epochs = {{choice([20])}} # Number of epochs
        batch_size = {{choice([32])}} # Batch size 
        lstm_dim = {{choice([64])}} # LSTM dimension
        emb_dim = {{choice([64])}} # Embedding dimension
        num_dnn = {{choice([1])}} # Size of outer DNN layers
        dense_dim = {{choice([64])}}
    
        
        input = Input(shape=(maxlen,))
        x = Embedding(input_dim=vocab_size,
                      output_dim=emb_dim,
                      input_length=maxlen,
                      mask_zero = True)(input)
        x = Bidirectional(LSTM(lstm_dim, return_sequences = False))(x)
        for i in range(num_dnn):
            x = Dense(dense_dim, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=[input], outputs=[output])
        #tensorboard = TensorBoard(log_dir=param)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        
        result = model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              #callbacks=[tensorboard],
              verbose=2,
              validation_data = (X_val, y_val))
            
        validation_acc = np.amax(result.history['val_acc']) 
    
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
    
if __name__ == '__main__':
    
    best_run, best_model = optim.minimize(model=LSTM,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())