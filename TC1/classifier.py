# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 15/10/19
# Natural Computing - 2019/2
# Computacional Assignment 1
# Classifier Implementation

from sklearn.neural_network import MLPClassifier

def Classifier(hidden_size, learning_rate, epochs, seed, verbose=False):
    model = MLPClassifier(hidden_layer_sizes=hidden_size, 
                          activation='relu',
                          solver='adam',
                          learning_rate_init=learning_rate,
                          max_iter=epochs,
                          random_state=seed,
                          verbose=verbose
                          )

    return model