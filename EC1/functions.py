
# -*- coding: utf-8 -*-

import numpy as np

# Implemented functions
# Ackley, Rosenbrock, Rastrigin, Langermann, Schwefel, Griewank


class Function():
    
    def __init__(self, dim):
        self.dimension = dim

    def dim(self):
        raise NotImplementedError
    
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

    def dim(self):
        return self.dimension
    
    def search_space(self):
        return (-32.768, 32.768)


class Rosenbrock(Function):
    
    def __call__(self):
        raise NotImplementedError

    def dim(self):
        raise NotImplementedError
    
    def search_space(self):
        raise NotImplementedError

class Rastrigin(Function):

    def __init__(self, dim=2):
        super().__init__(dim)
    
    def __call__(self, position):
        D = self.dim()

        s = np.square(position) - 10*np.cos(2*np.math.pi*position)

        return 10*D + s.sum()

    def dim(self):
        return self.dimension
    
    def search_space(self):
        return (-5.12, 5.12)

class Langermann(Function):
    
    def __init__(self, dim):
        super().__init__(dim)

    def dim(self):
        return self.dimension
    
    def search_space(self):
        raise NotImplementedError

class Schwefel(Function):

    def __init__(self, dim=2):
        super().__init__(dim)
    
    def __call__(self, position):
        A = 418.9829
        D = self.dim()
        s1 = np.sin(np.sqrt(np.abs(position))).sum()
        
        return A*D - s1

    def dim(self):
        return self.dimension
    
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
        s2 = np.cos(position/denominator).sum()

        return s1 - s2 + 1
    
    def dim(self):
        return self.dimension
    
    def search_space(self):
        return (-600, 600)