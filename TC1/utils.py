# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 15/10/19
# Natural Computing - 2019/2
# Computacional Assignment 1

class Fitness:
    def __init__(self, T, ro=0.9):
        self.total_features = T
        self.ro = ro
        self.phi = 1.0 - ro

    def __call__(self, error, features):
        return self.ro * error + self.phi * (features / self.total_features)

# def scale(scalar: float, vector: Vector) -> Vector:
#     return [scalar * num for num in vector]

def print_position(position):
    print("Best position: ", end="")
    print("[ ", end="")
    for i in range(len(position)):
        print("%.4f" % position[i], end="") # 4 decimals
        print(" ", end="")
    print("]")