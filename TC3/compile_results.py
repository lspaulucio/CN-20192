# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 03/12/19
# Natural Computing - 2019/2
# Computacional Assignment 3

import pickle
import numpy as np
from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Ionosphere, Wine, Arrhythmia
from sklearn.svm import SVC
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

def jaccard_score(sets):
    a = len(sets)
    pairs = [i for i in combinations(range(a), 2)]

    sets = [set(i[:7]) for i in sets]
    
    jac_score = []

    for i, j in pairs:
        intersection = sets[i].intersection(sets[j])
        union = sets[i].union(sets[j])
        jc = len(intersection) / len(union)
        jac_score.append(jc)

    return np.array(jac_score).mean()


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

    with open('results/{}_results.pickle'.format(args.dataset), 'rb') as f: 
        data = pickle.load(f) 

    importances = data['importances']
    times = np.array(data['times'])
    svm = np.array(data['svm_acc'])

    print("Dataset: {}".format(args.dataset))
    print('Execution time:')
    print("Mean: {:.3f}, Worst: {:.3f}, Better: {:.3f}".format(times.mean(), times.min(), times.max()))
    print('Classifier accuracy:')
    print("Mean: {:.3f}, Worst: {:.3f}, Better: {:.3f}".format(svm.mean(), svm.min(), svm.max()))
    print('Jaccard Score: {:.3f}\n'.format(jaccard_score(importances)))

    print(svm.max())
    svm = SVC(kernel='linear')
    importance = importances[np.argmax(svm)]
    mask = [0 if i not in importance[:NUM_SELECTED_FEATURES] else 1 for i in range(NUM_FEATURES)]
    x_train, y_train, x_test, y_test, classes = load_dataset()
    x, xt = x_train * mask, x_test * mask
    svm.fit(x, y_train)
    y_pred = svm.predict(xt)
    print(svm.score(xt, y_test))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, center=True, annot=True, cmap="Blues")
    
    plt.title("Matriz de Confusão")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()    
