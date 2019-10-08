# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 15/10/19
# Natural Computing - 2019/2
# Computacional Assignment 1

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def Arrhythmia():
    features = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", 
                "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"]
    
    arr = pd.read_csv('data/arrhythmia.arrhythmia.data', names=features)
    scaler = StandardScaler()
    x = arr.drop(columns=['Cultivator'])
    y = arr['Cultivator']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, y_train, x_test, y_test


def Ionosphere():
    features = ['Feature_{}'.format(str(i)) for i in range(34)]
    features += ['label'] # last column is the label
    label_map = {'g':0, 'b':1}
    ionosphere = pd.read_csv('data/ionosphere/ionosphere.data', names=features)
    ionosphere['label'] = ionosphere['label'].map(label_map)
    scaler = StandardScaler()
    x = ionosphere.drop(columns=['label'])
    y = ionosphere['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, y_train, x_test, y_test


def Wine():
    features = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", 
                "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"]
    wine = pd.read_csv('data/wine/wine.data', names=features)
    scaler = StandardScaler()
    x = wine.drop(columns=['Cultivator'])
    y = wine['Cultivator']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, y_train, x_test, y_test