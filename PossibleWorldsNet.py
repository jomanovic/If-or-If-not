# -*- coding: utf-8 -*-
"""
PossibleWorldsNet implementation from,
"Can Neural Networks Understand Logical Entailment"
https://openreview.net/pdf?id=SkZxCk-0Z

This model is originally based on: 
https://github.com/act65/entailment
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import led_parser
import utils
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Nullary(tf.keras.layers.Layer):
    def __init__(self, n_symbols, d_worlds, n_worlds, batch_size):
        super(Nullary, self).__init__()
        self.n_symbols = n_symbols
        self.d_worlds = d_worlds
        self.n_worlds = n_worlds
        self.batch_size = batch_size
        
        self.W = tf.Variable(tf.initializers.orthogonal()
                    (shape=(n_symbols, d_worlds, d_worlds)),
                    dtype=tf.float32,
                    name='Nullary_weights')

    def call(self, worlds, nullary):
        """
        worlds: "world" vectors of shape=(n_worlds, d_worlds)
        nullary: list of (batch_index, symbol_index)  
        """
        nullary = tf.constant(nullary, dtype=tf.int32)
        W_xs = tf.gather(self.W,nullary[:,1])
        x_s = tf.tensordot(W_xs, worlds, axes=[[2],[1]])
        x_s = tf.nn.l2_normalize(x_s, axis=1)
        
        return tf.scatter_nd(indices=tf.expand_dims(nullary[:,0], axis=-1),
                             updates=x_s,
                             shape=[self.batch_size, self.d_worlds, self.n_worlds])

class Unary(tf.keras.layers.Layer):
    def __init__(self, n_symbols, d_worlds, n_worlds, batch_size):
        super(Unary, self).__init__()
        self.n_symbols = n_symbols
        self.d_worlds = d_worlds
        self.n_worlds = n_worlds
        self.batch_size = batch_size

        self.W = tf.Variable(tf.initializers.orthogonal()
                    (shape=(n_symbols, d_worlds, d_worlds)),
                    dtype=tf.float32,
                    name='Unary_weights')

        self.b = tf.Variable(tf.initializers.orthogonal()
                    (shape=(n_symbols, d_worlds)),
                    dtype=tf.float32,
                    name='Unary_bias')
        
    def call(self, unary, computed_states):
        """
        unary: list of (batch_index, symbol_index, args_index)
        computed_states: list of previously computed states
        """
        indices, symbols, args = list(zip(*unary))
        args = np.array(args)
        
        # We stack previous batches ontop of each other.
        # To retrieve information computed at the k-th tree
        # during the i-th step we must go to index k*batch_size + i
        
        stacked_states = tf.concat(computed_states, axis=0)
        stacked_index = args[:,0]*self.batch_size + indices

        x_s = tf.gather(stacked_states, stacked_index)
        W = tf.gather(self.W, symbols)
        b = tf.gather(self.b, symbols)

        x = tf.matmul(W,x_s)
        x = tf.nn.l2_normalize(tf.add(x, tf.expand_dims(b,-1)),axis=1)
        
        return tf.scatter_nd(indices=tf.expand_dims(indices, axis=-1),
                             updates=x, 
                             shape=[self.batch_size, self.d_worlds, self.n_worlds])
    
    
class Binary(tf.keras.layers.Layer): 
    def __init__(self, n_symbols, d_worlds, n_worlds, batch_size):
        super(Binary, self).__init__()
        self.n_symbols = n_symbols
        self.d_worlds = d_worlds
        self.n_worlds = n_worlds
        self.batch_size = batch_size

        self.W = tf.Variable(tf.initializers.orthogonal()
                    (shape=(n_symbols, d_worlds, 2*d_worlds)),
                    dtype=tf.float32,
                    name='Binary_weights')

        self.b = tf.Variable(tf.initializers.orthogonal()
                    (shape=(n_symbols, d_worlds)),
                    dtype=tf.float32,
                    name='Binary_bias')
        
    def call(self, binary, computed_states):
        """
        binary: list of (batch_index, symbol_index, args_index)
        computed_states: list of previously computed states
        """
        indices, symbols, args = zip(*binary)
        args = np.array(args)
        stacked_states = tf.concat(computed_states, axis=0)

        l_idx = args[:, 0]*self.batch_size + indices
        r_idx = args[:, 1]*self.batch_size + indices
        l = tf.gather(stacked_states, l_idx)
        r = tf.gather(stacked_states, r_idx)
        x = tf.concat([l, r], axis=1) 

        # [ n_symbols x 2*d_worlds x d_worlds]
        W = tf.gather(self.W, symbols)
        b = tf.gather(self.b, symbols)

        x_s = tf.matmul(W, x)
        x_s = tf.nn.l2_normalize(x_s + tf.expand_dims(b, -1), axis=1)

        
        return tf.scatter_nd(indices=tf.expand_dims(indices, axis=-1)  ,
                             updates=x_s,
                             shape=[self.batch_size, self.d_worlds, self.n_worlds])


class Sat3Cell(tf.keras.layers.Layer):
    def __init__(self, n_symbols, d_worlds, n_worlds, batch_size):
        super(Sat3Cell, self).__init__()
        self.n_symbols = n_symbols
        self.d_worlds = d_worlds
        self.n_worlds = n_worlds
        self.batch_size = batch_size        

        self.Nullary = Nullary(n_symbols, d_worlds, n_worlds, batch_size)
        self.Unary = Unary(n_symbols, d_worlds, n_worlds, batch_size)
        self.Binary = Binary(n_symbols, d_worlds, n_worlds, batch_size)
        
        
    def call(self, worlds, batch_data, computed_states):
        """
        worlds: "world" vectors of shape=(n_worlds, d_worlds)
        batch_data: list of (batch_index, symbol_index, args_index)
        computed_states: list of previously computed states
        """
        nullary, unary, binary = [], [], []
        indices, symbols, args = zip(*batch_data)
        
        for i in range(len(indices)):
            if len(args[i]) == 0:
                nullary.append((indices[i], symbols[i]))
            elif len(args[i]) == 1:
                unary.append((indices[i], symbols[i], args[i]))
            elif len(args[i]) == 2:
                binary.append((indices[i], symbols[i], args[i]))
        
        batch_output = tf.zeros([self.batch_size, self.d_worlds, self.n_worlds], dtype=tf.float32)
        
        if nullary:
            batch_output += self.Nullary(worlds, nullary)
        if unary:
            batch_output += self.Unary(unary, computed_states)
        if binary:
            batch_output += self.Binary(binary, computed_states)
        
        return batch_output

        
class TreeNets(tf.keras.layers.Layer):
    def __init__(self, cell, n_worlds, batch_size):
        super(TreeNets, self).__init__()
        self.cell = cell
        self.n_worlds = n_worlds
        self.batch_size = batch_size
    
    def call(self, worlds, trees):
        """
        worlds: "world" vectors of shape=(n_worlds, d_worlds)
        trees: list of parsed sentences
        """
        computed_states = []
        computed_index = []

        max_len_tree = len(max(trees, key=lambda x: len(x[0]))[0])
        
        for s_index in range(max_len_tree):
            batch_data = [] 
            for t_index, tree in enumerate(trees):
                symbols, args = tree
                if s_index < len(symbols):
                    new_args = [arg + s_index for arg in args[s_index]]
                    batch_data.append((t_index, symbols[s_index], new_args))
                    
                    # This condition ensures the computation is complete
                    # for the symbol at s_index of tree at t_index
                    if s_index == len(symbols)-1:
                        computed_index.append((s_index,t_index))
                        
            state = self.cell(worlds, batch_data, computed_states)
            computed_states.append(state)
        
        computed_index = tf.constant(sorted(computed_index, key=lambda x: x[1]))
        return tf.gather_nd(tf.stack(computed_states),computed_index)
        
class PossibleWorldsNet(tf.keras.layers.Layer):
    
    def __init__(self, parser, encoder, n_worlds, d_worlds):
        super(PossibleWorldsNet, self).__init__()
        self.parser = parser
        self.encoder = encoder
        self.n_worlds = n_worlds
        self.d_worlds = d_worlds
        
        self.worlds = tf.Variable(tf.initializers.orthogonal()
                    (shape=(n_worlds, d_worlds)),
                    dtype=tf.float32,
                    trainable=False,
                    name='worlds')
        
        self.W = tf.Variable(tf.initializers.orthogonal()
                            (shape=(2*d_worlds, 1)),
                            dtype=tf.float32,
                            name='linear_weight')
        
        self.b = tf.Variable(tf.initializers.constant(0.)
                            (shape=(1,1)),
                             dtype=tf.float32,
                             name='linear_bias')
        
    def call(self, A, B):
        """
        A, B: lists of logical formulae i.e. ['(x OR y) AND z']
        """
        A_trees = [self.parser.parse(sentence) for sentence in A]
        B_trees = [self.parser.parse(sentence) for sentence in B]
        
        A_encoded = self.encoder(self.worlds, A_trees)
        B_encoded = self.encoder(self.worlds, B_trees)        
        x = tf.concat([A_encoded, B_encoded], axis=1)

        p = tf.tensordot(x, self.W, axes=[[1],[0]])
        p = tf.reshape(p, [self.encoder.batch_size, self.n_worlds]) + self.b
        p = tf.math.reduce_prod(tf.nn.sigmoid(p), axis=1)
        
        return p


""" 
TRAINING PHASE:
"""

cross_entropy = tf.keras.losses.BinaryCrossentropy()
accuracy = tf.keras.metrics.BinaryAccuracy()

def compute_step(model, A, B, y):
    with tf.GradientTape() as tape:
        y_pred = model(A, B)
        loss = cross_entropy(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)

    for g, v in zip(grads, model.trainable_variables):
        if g is None:
            raise ValueError('No gradient for {}'.format(v.name))

    return loss, grads, y


if __name__ == "__main__":
    
    d_worlds = 64
    n_worlds = 32
    batch_size = 32
    n_epochs = 30
    
    language = led_parser.propositional_language()
    parser = led_parser.Parser(language)
    n_symbols = len(language.symbols)

    cell = Sat3Cell(n_symbols, d_worlds, n_worlds, batch_size)
    encoder = TreeNets(cell, n_worlds, batch_size)
    PWN = PossibleWorldsNet(parser, encoder, n_worlds, d_worlds)
    optimizer = tf.keras.optimizers.Adam()

    for i in range(n_epochs):
        print(i)
        for A, B, y in utils.batch_data(utils.read_data('../Logical_Entailment/temp/train.txt'), batch_size):
            loss, grads, p = compute_step(PWN, A, B, y)
            gradients = zip(grads, PWN.trainable_variables)
            optimizer.apply_gradients(gradients)
            logging.info('loss: {}'.format(tf.reduce_mean(loss)))

    
        for A, B, y in utils.batch_data(utils.read_data('../Logical_Entailment/temp/validate.txt'), batch_size):

            acc = np.mean([accuracy(y, PWN(A, B))])
            logging.info('accuracy: {}'.format(acc))
