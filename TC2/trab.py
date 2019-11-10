# -*- coding: utf-8 -*-
"""
Aluno: Leonardo Santos Paulucio
Data: 15/11/19
Computação Natural - Trabalho Computacional 2

"""

from sklearn.datasets import load_iris

def normalize(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# Importing iris dataset
iris = load_iris()

w1 = 0.5
w2 = 0.5
w3 = 0.5
w4 = 0.5

inp = np.array([0.22222222, 0.625     , 0.06779661, 0.04166667]) #setosa
inp = np.array([0.38888889, 0.33333333, 0.59322034, 0.5       ]) # versicolor
inp = np.array([0.91666667, 0.41666667, 0.94915254, 0.83333333]) #virginica