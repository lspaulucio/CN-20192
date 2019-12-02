# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 03/12/19
# Natural Computing - 2019/2
# Computacional Assignment 3

import numpy as np
from time import time
import pickle
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from datasets import Ionosphere, Wine, Arrhythmia
import argparse

def sum_weights(weights):
    n = len(weights)
    shape = weights[0].shape
    s = np.zeros(shape)
    
    for i in range(n):
        s += weights[i]

    return np.abs(s.sum(axis=1))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Trabalho 3')
    parser.add_argument('--dataset', action='store', dest='dataset', required=True, choices=['wine', 'ionosphere', 'arrhythmia'], help='Dataset que sera usado')

    args = parser.parse_args()

    if args.dataset == 'wine':
        load_dataset = Wine
        NUM_FEATURES = 13
        NUM_SELECTED_FEATURES = 7
        HIDDEN_SIZE = 7
    elif args.dataset == 'ionosphere':
        load_dataset = Ionosphere
        NUM_FEATURES = 34
        NUM_SELECTED_FEATURES = 19
        HIDDEN_SIZE = 9
    elif args.dataset == 'arrhythmia':
        load_dataset = Arrhythmia
        NUM_FEATURES = 279
        NUM_SELECTED_FEATURES = 132
        HIDDEN_SIZE = 67

    NUM_ITER = 20
    NUM_EPOCHS = 100
    RUNS = 6

    importances = []
    svm_acc = []
    times = []

    for j in range(RUNS):
        start_time = time()
        best_weights_list = []

        for ITER in range(NUM_ITER):
            x_train, y_train, x_test, y_test, classes = load_dataset()
            net = MLPClassifier(hidden_layer_sizes=(HIDDEN_SIZE))

            best_weights = []
            best_acc = 0

            for i in range(NUM_EPOCHS):
                net.partial_fit(x_train, y_train, classes=classes)
                weights = net.coefs_[0]
                val_acc = net.score(x_test, y_test)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_weights = weights
            
            best_weights_list.append(best_weights)
            print(best_acc)
        
        summed_weights = sum_weights(best_weights_list)
        importance = list(reversed(np.argsort(summed_weights))) 
        importances.append(importance)

        svm = SVC(kernel='linear')
        mask = [0 if i not in importance[:NUM_SELECTED_FEATURES] else 1 for i in range(NUM_FEATURES)]
        x, xt = x_train * mask, x_test * mask
        svm.fit(x, y_train)
        score = svm.score(xt, y_test)
        
        svm_acc.append(score)
        times.append(time() - start_time)

    print(svm_acc)

    results = {'importances': importances,
               'svm_acc': svm_acc,
               'times': times}

    with open('{}_results.pickle'.format(args.dataset), 'wb') as f:
        pickle.dump(results, f)
