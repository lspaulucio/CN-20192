# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 03/12/19
# Natural Computing - 2019/2
# Computacional Assignment 3

import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix
from datasets import Ionosphere, Wine, Arrhythmia
from time import time

def sum_weights(weights):
    n = len(weights)
    shape = weights[0].shape
    s = np.zeros(shape)
    
    for i in range(n):
        s += weights[i]

    return np.abs(s.sum(axis=1))

if __name__ == "__main__":

    NUM_ITER = 20
    NUM_EPOCHS = 100
    RUNS = 6

    # Adicionar time
    importances = []
    svm_acc = []
    for j in range(RUNS):
        best_weights_list = []

        for ITER in range(NUM_ITER):
            x_train, y_train, x_test, y_test, classes = Wine()
            num_features = x_train.shape[1]
            net = MLPClassifier(hidden_layer_sizes=(num_features))

            best_weights = []
            best_acc = 0

            for i in range(NUM_EPOCHS):
                net.partial_fit(x_train, y_train, classes=classes)
                weights = net.coefs_[0]
                val_acc = net.score(x_test, y_test)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_weights = weights
                    # print(best_acc)
            
            best_weights_list.append(best_weights)
            print(best_acc)
        
        summed_weights = sum_weights(best_weights_list)
        importance = list(reversed(np.argsort(summed_weights))) 
        importances.append(importance)

        svm = SVC(kernel='linear')
        NUM_SELECTED_FEATURES = 7
        mask = [0 if i not in importance[:NUM_SELECTED_FEATURES] else 1 for i in range(num_features)]
        x, xt = x_train * mask, x_test * mask
        svm.fit(x, y_train)
        score = svm.score(xt, y_test)
        svm_acc.append(score)
        
    print(svm_acc)




#
# wine = array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1])
# features = 0, 3, 5, 8, 9, 10, 12 - 7
# Ionosphere 
# array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
# array([0, 1, 2, 4, 6, 7, 9, 10, 11, 13, 15, 17, 18, 22, 23, 27, 28, 30, 33]) 19

# Arrithmia 132
# [1, 2, 6, 8, 9, 12, 13, 15, 20, 21, 23, 25, 26, 27, 29, 30, 31, 35, 37, 39, 44, 45, 50, 51, 55, 57, 62, 63, 64, 65, 67, 72, 74, 75, 76, 77,
#  80, 81,86, 89, 90, 91,92, 94, 96, 97, 101, 102, 104, 105, 108, 109, 110, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 128, 129,
#  130, 131, 135, 136, 138, 139, 141, 142, 143, 146, 147, 148, 150, 156, 157, 158, 159, 160, 164, 166, 177, 178, 179, 181, 185, 186, 188, 191,
#  195, 196, 197, 198, 199, 200, 206, 207, 210, 211, 216, 217, 220, 221, 222, 225, 226, 229, 231, 233, 235, 239, 241, 242, 244, 249, 250, 251,
#  260, 261, 264,265, 266, 267, 271, 272, 277, 278]
