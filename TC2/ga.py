# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 17/09/19
# Natural Computing - 2019/2
# Computacional Exercise 1

# Genetic Algorithm implementation

import numpy as np
from copy import deepcopy

class Individual:
    def __init__(self, dim, minx=-10, maxx=10, position=None):
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.score = 0
        
        if position is None:
            self.position = (maxx - minx) * np.random.rand(dim) + minx
        else:
            position[position < minx] = minx
            position[position > maxx] = maxx
            self.position = position

        self.error = float('inf')  # initial error

    def mutate(self, sigma):
        i = np.random.randint(self.dim)
        gene = self.position[i]
        beta = np.random.normal(gene, sigma)
        new_gene = beta * self.position[i]
        if new_gene < 0:
            new_gene = 0
        elif new_gene > 1:
            new_gene = 1
        self.position[i] = new_gene

    def cross(self, other, beta):
        new_position = (beta * self.position) + (1 - beta) * other.position
        new = Individual(self.dim, self.minx, self.maxx, position=new_position)
        return new

    def evaluate(self, fitness):
        self.error = fitness(self.position)


class Population:
    def __init__(self, size, dimension, min, max, crossover_prob=0.65, mutation_prob=0.1, sel_frac=0.2):
        minx, maxx = min, max
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.dim = dimension
        self.size = size
        self.population = [Individual(dimension, minx, maxx) for _ in range(size)]
        self.parents = []
        self.offspring = []
        self.best_individual = self.population[0]
        self.best_error = self.population[0].error
        self.selection_fraction = sel_frac        
        self.evaluate()

    def evaluate(self):
        for ind in self.population:
            if ind.error < self.best_error:
                self.best_error = deepcopy(ind.error)
                self.best_individual = deepcopy(ind)

    def selection(self):

        num_parents = int(self.selection_fraction * self.size)        

        parents = []
        
        for _ in range(num_parents):
            sample = list(np.random.choice(self.population, 2, replace=False))
            if sample[0].error < sample[1].error:
                parents.append(sample[0])
            else:
                parents.append(sample[1])

        self.parents = parents

    
    def crossover(self):

        new_offspring = []

        for _ in range(self.size // 2):
            prob = np.random.rand()
            sample = list(np.random.choice(self.parents, 2, replace=False))
            if prob <= self.crossover_prob:
                beta = np.random.rand()
                ind1 = sample[0].cross(sample[1], beta)
                ind2 = sample[1].cross(sample[0], beta)
            else:
                ind1 = sample[0]
                ind2 = sample[1]
            
            new_offspring.append(ind1)
            new_offspring.append(ind2)

        self.offspring = new_offspring
        
    def mutation(self, sigma=1):
        for ind in self.offspring:
            prob = np.random.rand()
            if prob <= self.mutation_prob:
                ind.mutate(sigma)

    def new_generation(self):
        self.selection()
        self.crossover()
        self.mutation(sigma=0.3)
        self.population = self.offspring

    def __iter__(self):
        self.sentinel = 0
        return self

    def __next__(self):
        if self.sentinel < self.size:
            ind = self.population[self.sentinel]
            self.sentinel += 1
            return ind
        else:
            raise StopIteration
        