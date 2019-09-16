
# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 17/09/19
# Natural Computing - 2019/2
# Computacional Exercise 1

# Functions for optimization
# Implemented functions: 
# Ackley, Rosenbrock, Rastrigin, Langermann, Schwefel, Griewank

import numpy as np

class Function():
    
    def __init__(self, dim):
        self.dimension = dim

    def dim(self):
        return self.dimension
    
    def __call__(self):
        raise NotImplementedError

    def search_space(self):
        raise NotImplementedError

class Ackley(Function):
    
    def __init__(self, dim=2):
        super().__init__(dim)

    def __call__(self, X):
        A = 20
        B = 0.2
        C = 2 * np.math.pi
        s1 = np.square(X).mean()
        s2 = np.cos(C * X).mean()
        return -A * np.exp(-B * np.sqrt(s1)) - np.exp(s2) + A + np.exp(1)
    
    def search_space(self):
        return (-32.768, 32.768)


class Rosenbrock(Function):
    
    def __init__(self, dim=2):
        super().__init__(dim)

    def __call__(self, X):
        A = 100
        D = self.dim()
        sum = 0
        for i in range(D-1):
            x_i = X[i]
            x_next = X[i+1]
            sum += A*(x_next - x_i**2)**2 + (x_i - 1)**2

        return sum

    def search_space(self):
        return (-2.048, 2.048)

class Rastrigin(Function):

    def __init__(self, dim=2):
        super().__init__(dim)
    
    def __call__(self, X, A=10):
        D = self.dim()
        return A*D + np.sum(X**2 - A * np.cos(2 * np.math.pi * X))
    
    def search_space(self):
        return (-5.12, 5.12)

class Langermann(Function):
    
    def __init__(self, dim):
        super().__init__(dim)
        assert (dim == 2)
    
    def __call__(self, X):
        A = np.array([[3, 5],
                      [5, 2],
                      [2, 1],
                      [1, 4],
                      [7, 9]])

        C = np.array([1, 2, 5, 2, 3])
        M = 5
        D = self.dim()
        
        outer = 0
        for i in range(M):
            inner = 0
            for j in range(D):
                x_j = X[j]
                inner += (x_j - A[i][j])**2
            new = C[i] * np.exp(-inner/np.math.pi) * np.cos(np.math.pi*inner)
            outer = outer + new
        
        return -outer


    def search_space(self):
        return (0, 10)

class Schwefel(Function):

    def __init__(self, dim=2):
        super().__init__(dim)
    
    def __call__(self, X):
        A = 418.9829
        D = self.dim()
        s1 = np.sum(X * np.sin(np.sqrt(np.abs(X))))
        
        return A*D - s1
    
    def search_space(self):
        return (-500, 500)

class Griewank(Function):
    
    def __init__(self, dim=2):
        super().__init__(dim)

    def __call__(self, X):
        A = 4000
        s1 = np.square(X).sum() / A
        denominator = [np.sqrt(i) for i in range(1, self.dim()+1)]
        denominator = np.array(denominator)
        s2 = np.prod(np.cos(X / denominator))

        return s1 - s2 + 1
        
    def search_space(self):
        return (-600, 600)

if __name__ == "__main__":

    f = Rastrigin(2)
    X = np.array([0,0])
    print(f(X))