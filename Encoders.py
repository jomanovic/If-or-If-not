# -*- coding: utf-8 -*-
"""
We implement efficent ConvNet, LSTM and BidirLSTM models.
Hyperparameter optimization is performed with,
Hyperopt + Keras : https://github.com/maxpumperla/hyperas
"""


# Hyper-parameter optimization
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from tensorflow.keras.layers import LSTM, CuDNNLSTM, Flatten, Reshape, concatenate, TimeDistributed
from tensorflow.keras.layers import Dropout, Input, Dense, ThresholdedReLU, Embedding, Bidirectional 
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import TensorBoard

from utils import vocabulary, to_numeric

import tensorflow as tf
import numpy as np

def data():
    
    
    A_0, B_0, y_0 = zip(*data.read_data('../Logical_Entailment/temp/train.txt'))
    A_1, B_1, y_1 = zip(*data.read_data('../Logical_Entailment/temp/validate.txt'))
    y_0, y_1 = np.array(y_0), np.array(y_1)
    
    # Longest sentence length
    len_A = len(max(*A_0, key=len))
    len_B = len(max(*B_0, key=len))
    len_X = max(len_A, len_B)
    
    # Vocabulary
    
    vocabulary = data.vocabulary(A_0+B_0)
    
    # Ensure sentences are of same length 
    
    train_A = pad_sequences(data.num_encode(A_0, vocabulary),
                          padding='post',
                          maxlen=len_X)
    
    train_B = pad_sequences(data.num_encode(B_0, vocabulary),
                          padding='post',
                          maxlen=len_X)
    
    validate_A = pad_sequences(data.num_encode(A_1, vocabulary),
                              padding='post',
                              maxlen=len_X)
    
    validate_B = pad_sequences(data.num_encode(B_1, vocabulary),
                              padding='post',
                              maxlen=len_X)

    
    return train_A, train_B, y_0, validate_A, validate_B, y_1, len_X, len(vocabulary)
    
def LSTM(A, B, y, val_A, val_B, val_y, maxlen, vocab_size):
    
    with tf.device('/GPU:0'):
        
        lr = {{choice([1e-3, 1e-4])}} # Learning rate
        epochs = 40 # Number of epochs
        batch_size = 32 # Batch size 
        lstm_dim = 64 # LSTM dimension
        emb_dim = 32 # Embedding dimension
        num_dnn = 1 # Size of outer DNN layers
        dense_dim = 64

        
        input_aux = Input(shape=(maxlen,))
        x = Embedding(input_dim=vocab_size,
                      output_dim=emb_dim,
                      input_length=maxlen)(input_aux)  
        output_aux = CuDNNLSTM(lstm_dim, return_sequences = True)(x)
        model_aux = Model(inputs=[input_aux], outputs=[output_aux])
        
        # Creating outer model 
        
        input_1 = Input(shape=(maxlen,))
        input_2 = Input(shape=(maxlen,))
        
        x_1 = model_aux(input_1)
        x_2 = model_aux(input_2)
        x = concatenate([x_1, x_2])
        
        for _ in range(num_dnn):
            x = TimeDistributed(Dense(dense_dim, activation='relu'))(x)
        x = Flatten()(x)
        output = Dense(1, activation='sigmoid')(x)
                
        model = Model(inputs=[input_1,input_2], outputs=[output])
        #tensorboard = TensorBoard(log_dir=param)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        
        result = model.fit([X_1, X_2], y,
              batch_size=batch_size,
              epochs=epochs,
              #callbacks=[tensorboard],
              verbose=2,
              validation_data = ([X_1_val, X_2_val], y_val))
            
        validation_acc = np.amax(result.history['val_acc']) 
        print('Best validation acc of epoch:', validation_acc)
        
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
    
def biLSTM(X_1, X_2, y, X_1_val, X_2_val, y_val, maxlen, vocab_size):
    # best so far:
    # lr-0.001-epochs-10-batch_size-64-lstm_dim-32-dense_dim-64-emb_dim-32-num_dnn-1
    # lr-0.001-epochs-20-batch_size-32-lstm_dim-64-dense_dim-64-emb_dim-64-num_dnn-2
    tf.keras.backend.clear_session()
    with tf.device('/GPU:0'):
        
        lr = {{choice([1e-3, 1e-4])}} # Learning rate
        epochs = {{choice([20])}} # Number of epochs
        batch_size = {{choice([32, 64])}} # Batch size 
        lstm_dim = {{choice([32, 64])}} # LSTM dimension
        emb_dim = {{choice([32, 64])}} # Embedding dimension
        num_dnn = {{choice([1, 2])}} # Size of outer DNN layers

        
        input_aux = Input(shape=(maxlen,))
        x = Embedding(input_dim=vocab_size,
                      output_dim=emb_dim,
                      input_length=maxlen)(input_aux)
        
        #output_aux_1 = CuDNNLSTM(lstm_dim, go_backwards=False)(x)
        #output_aux_2 = CuDNNLSTM(lstm_dim)(x)
        #output_aux = concatenate([output_aux_1, output_aux_2])
        output_aux = Bidirectional(CuDNNLSTM(lstm_dim))(x)
        model_aux = Model(inputs=[input_aux], outputs=[output_aux])
        
        # Creating outer model 
        
        input_1 = Input(shape=(maxlen,))
        input_2 = Input(shape=(maxlen,))
        
        x_1 = model_aux(input_1)
        x_2 = model_aux(input_2)
        x = concatenate([x_1, x_2])
        
        for i in range(num_dnn):
            dense_dim = {{choice([32, 64])}}
            x = Dense(dense_dim, activation='relu')(x)
        x = Flatten()(x)
        output = Dense(1, activation='sigmoid')(x)
                
        model = Model(inputs=[input_1,input_2], outputs=[output])
        #tensorboard = TensorBoard(log_dir=param)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                loss='binary_crossentropy',
                metrics=['accuracy'])
        
        result = model.fit([X_1, X_2], y,
              batch_size=batch_size,
              epochs=epochs,
              #callbacks=[tensorboard],
              verbose=2,
              validation_data = ([X_1_val, X_2_val], y_val))
            
        validation_acc = np.amax(result.history['val_acc']) 
        
        print(param)
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

    
def CNN(X_1, X_2, y, X_1_val, X_2_val, y_val, maxlen, vocab_size):
    
    tf.keras.backend.clear_session()

    lr = {{choice([1e-2, 1e-3,1e-4,1e-5])}} # Learning rate
    epochs = {{choice([10, 20, 30])}} # Number of epochs
    batch_size = {{choice([64, 128])}} # Batch size 
    emb_dim = {{choice([32, 64])}} # Embedding dimension
    num_cnn = {{choice([4, 6, 8])}} # Number of CNN layers
    inter_pool = {{choice([0, 1, 3, 5])}} # Number of intermittent POOLing layers
    num_inner_dnn = {{choice([2, 3])}} # Size of inner DNN layers
    num_outer_dnn = {{choice([1, 2])}} # Size of outer DNN layers

    print('Build model...')
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)
    
    # Creating inner model
    
    input_aux = Input(shape=(maxlen,))
    
    emb = Embedding(input_dim=vocab_size,
                  output_dim=emb_dim,
                  input_length=maxlen)
    
    x = emb(input_aux)
    
    filters = []
    kernel_size = []
    pool_size = []
    
    input_length = maxlen
    works = True
    
    for i in range(num_cnn):
        filter_choice = {{choice([32,64])}}
        kernel_choice = {{choice([5,7,9])}}
        
        while input_length - kernel_choice  + 1 <= 0: 
            if kernel_choice >= 5:
                kernel_choice -= 2
            else:
                works = False
                break

        if works == False:
            break 
        
        input_length = input_length - kernel_choice  + 1
        print('input_length:' + str(input_length))
        print('kernel_size:' + str(kernel_choice))
        filters.append(filter_choice)
        kernel_size.append(kernel_choice)
        
        x = Conv1D(filters=filters[-1],
                   kernel_size=kernel_size[-1],
                   strides=1,
                   padding='valid',
                   kernel_initializer=initializer,
                   activation='tanh')(x)
        
        if i>0 and inter_pool>0 and (i+1)%inter_pool == 0:
            print('input_length pool bfr:' + str(input_length))
            input_length = int(input_length/3)

            if input_length  <= 0:
                break 
            x = MaxPool1D(pool_size=3,
                          padding='valid')(x)
            print('input_length pool aftr:' + str(input_length))
    x = Flatten()(x)
    for _ in range(num_inner_dnn):
        x = Dense({{choice([32, 64])}}, activation='relu')(x)
        x = Dropout(0.5)(x)
    output_aux = x    
    
    model_aux = Model(inputs=input_aux, outputs=output_aux)
    
    # Creating outer model 
    
    input_1 = Input(shape=(maxlen,))
    input_2 = Input(shape=(maxlen,))
    
    x_1 = model_aux(input_1)
    x_2 = model_aux(input_2)
    x = concatenate([x_1, x_2])
    
    for _ in range(num_outer_dnn):
        x = Dense({{choice([32, 64])}}, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_1,input_2], outputs=output)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    result = model.fit([X_1, X_2], y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data = ([X_1_val, X_2_val], y_val))
    
    
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
    
if __name__ == '__main__':
    
    best_run, best_model = optim.minimize(model=LSTM,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    





