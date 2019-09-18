import numpy as np
import numpy.linalg as la

from utils import print_position
import functions as F

"""
The Algorithm

Paramters
Population size  : N
Covariance Matrix: C
mean of best k   : mu

1. Sample N from multivariate normal distribution
2. Calculate fitness and 

Based on https://github.com/jenkspt/CMA-ES

"""

def CMA(max_epochs, population_size, func_fitness, elite_size=25, sigma0=0.5):
    low, high = func_fitness.search_space()

    d = func_fitness.dim()     # Dimensions
    n = population_size   # Population size
    k = elite_size        # Size of elite population

    X = np.random.normal(0, 2, (d, n))

    for i in range(max_epochs):
        # Minimize this function
        fitness = func_fitness(X)
        arg_topk = np.argsort(fitness)[:k]
        topk = X[:, arg_topk]
        print('Iter {}, score {}, X = {}'.format(i, fitness[arg_topk[0]], X[:,arg_topk[0]]))
        # Covariance of topk but using mean of entire population
        X[np.isnan(X)] = 0 
        centered = topk - X.mean(1, keepdims=True)
        C = (centered @ centered.T)/(k-1)
        C[np.isnan(C)] = 0 
        # Eigenvalue decomposition
        w, E = la.eigh(C)
        # Generate new population
        # Sample from multivariate gaussian with mean of topk
        N = np.random.normal(size=(d, n))
        X = topk.mean(1, keepdims=True) + (E @ np.diag(np.sqrt(w)) @ N)

    return X[:, arg_topk[0]]

if __name__ == "__main__":

    fitness = F.Schwefel(2)

    epochs = 100

    best_position = CMA(epochs, 200, fitness)

    print("Solution found after {} epochs".format(epochs))
    print_position(best_position)
    print("Best error: {:.3f}".format(fitness(best_position)))
