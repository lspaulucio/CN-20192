# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 15/10/19
# Natural Computing - 2019/2
# Computacional Assignment 1
# Classifier Implementation

import numpy as np
from sklearn.neural_network import MLPClassifier

def Classifier(hidden_size, learning_rate, epochs, verbose=False):
    model = MLPClassifier(hidden_layer_sizes=hidden_size, 
                          activation='relu',
                          solver='adam',
                          learning_rate_init=learning_rate,
                          max_iter=epochs,
                          verbose=verbose
                          )

    return model


# Based on Iván Vallés Pérez implementation available on https://github.com/ivallesp/simplestELM

class ELM():
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units

    def fit(self, X, labels):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        self.random_weights = np.random.randn(X.shape[1], self.n_hidden_units)
        G = np.tanh(X.dot(self.random_weights))
        self.w_elm = np.linalg.pinv(G).dot(labels)

    def predict(self, X):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        G = np.tanh(X.dot(self.random_weights))
        return G.dot(self.w_elm)