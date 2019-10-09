# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 15/10/19
# Natural Computing - 2019/2
# Computacional Assignment 1

import numpy as np
import random
from tqdm import tqdm

from classifier import Classifier
from datasets import  Arrhythmia, Ionosphere, Wine

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Setting seed for reproducibility
SEED = 500
random.seed(SEED)
np.random.seed(SEED)

def classifierArrhythmia():
    
    # Hyperparameters
    NUM_EPOCHS = 500
    NUM_FEATURES = 279
    NUM_CLASSES = 16
    HIDDEN_SIZE = int(np.sqrt(NUM_FEATURES * NUM_CLASSES))
    LEARNING_RATE = 1e-3

    x_train, y_train, x_test, y_test = Arrhythmia()

    model = Classifier(HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS, verbose=True)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred, digits=4))
    sns.heatmap(cm, center=True, annot=True, cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()    
    print(model.score(x_test, y_test))


def classifierIonosphere():

    # Hyperparameters
    NUM_EPOCHS = 500
    NUM_FEATURES = 34
    NUM_CLASSES = 2
    HIDDEN_SIZE = int(np.sqrt(NUM_FEATURES * NUM_CLASSES))
    LEARNING_RATE = 1e-3

    x_train, y_train, x_test, y_test = Ionosphere()

    model = Classifier(HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS, verbose=True)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred, digits=4))
    sns.heatmap(cm, center=True, annot=True, cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()    


def classifierWine():
    
    # Hyperparameters
    NUM_EPOCHS = 500
    NUM_FEATURES = 13
    NUM_CLASSES = 3
    HIDDEN_SIZE = int(np.sqrt(NUM_FEATURES * NUM_CLASSES))
    LEARNING_RATE = 1e-3

    x_train, y_train, x_test, y_test = Wine()

    model = Classifier(HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS, verbose=True)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred, digits=4))
    sns.heatmap(cm, center=True, annot=True, cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == "__main__":

    classifierArrhythmia()
    classifierIonosphere()
    classifierWine()
