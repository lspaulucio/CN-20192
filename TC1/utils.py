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


def sampleFromClass(ds, k):
    class_counts = {i:0 for i in range(len(k))}
    all_ids = set(range(len(ds)))
    test_ids = set()
    
    for i, (_, label) in enumerate(ds):
        c = label
        if class_counts[c] <= k[c]: 
            class_counts[c] = class_counts.get(c,0) + 1 
            test_ids.add(i) 

    train_ids = list(all_ids - test_ids)
    train_ids.sort()
    test_ids = list(test_ids)
    test_ids.sort()
    print("Generating subsets")
    return Subset(ds, train_ids), Subset(ds, test_ids)