
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

    def __call__(self, position):
        A = 20
        B = 0.2
        C = 2 * np.math.pi
        s1 = np.square(position).mean()
        s2 = np.cos(C * position).mean()
        return -A * np.exp(-B * np.sqrt(s1)) - np.exp(s2) + A + np.exp(1)
    
    def search_space(self):
        return (-32.768, 32.768)


class Rosenbrock(Function):
    
    def __init__(self, dim=2):
        super().__init__(dim)

    def __call__(self, position):
        A = 100
        D = self.dim()
        sum = 0
        for i in range(D-1):
            x_i = position[i]
            x_next = position[i+1]
            sum += A*(x_next - x_i**2)**2 + (x_i - 1)**2

        return sum

    def search_space(self):
        return (-2.048, 2.048)

class Rastrigin(Function):

    def __init__(self, dim=2):
        super().__init__(dim)
    
    def __call__(self, position):
        D = self.dim()
        s = np.square(position) - 10*np.cos(2*np.math.pi*position)
        return 10*D + s.sum()
    
    def search_space(self):
        return (-5.12, 5.12)

class Langermann(Function):
    
    def __init__(self, dim):
        super().__init__(dim)
        assert (dim == 2)
    
    def __call__(self, position):
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
                x_j = position[j]
                inner += (x_j - A[i][j])**2
            new = C[i] * np.exp(-inner/np.math.pi) * np.cos(np.math.pi*inner)
            outer = outer + new
        
        return -outer


    def search_space(self):
        return (0, 10)

class Schwefel(Function):

    def __init__(self, dim=2):
        super().__init__(dim)
    
    def __call__(self, position):
        A = 418.9829
        D = self.dim()
        s1 = np.sum(position * np.sin(np.sqrt(np.abs(position))))
        
        return A*D - s1
    
    def search_space(self):
        return (-500, 500)

class Griewank(Function):
    
    def __init__(self, dim=2):
        super().__init__(dim)

    def __call__(self, position):
        A = 4000
        s1 = np.square(position).sum() / A
        denominator = [np.sqrt(i) for i in range(1, self.dim()+1)]
        denominator = np.array(denominator)
        s2 = np.prod(np.cos(position / denominator))

        return s1 - s2 + 1
        
    def search_space(self):
        return (-600, 600)