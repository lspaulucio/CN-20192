# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 15/10/19
# Natural Computing - 2019/2
# Computacional Assignment 1
# Genetic Algorithm implementation

import numpy as np
from utils import print_position

class Individual:
    def __init__(self, dim):
        self.dim = dim
        self.error = float('inf')  # initial error
        self.features = 0       
        self.chromosome = np.random.randint(low=0, high=2, size=dim)
        self.update_features()

    def update_features(self):
        self.features = (self.chromosome == 1).sum()

    def cross(self, other, cross_prob):
        pos = np.random.randint(0, self.dim) # cut point
        new = Individual(self.dim)
        new.chromosome[pos:] = other.chromosome[pos:]
        new.update_features()
        return new

    def mutate(self):
        point = np.random.randint(self.dim)
        self.chromosome[point] ^= 1 # gene xor 1 - to invert bit
        self.update_features()

    def evaluate(self, fitness, error):
        self.error = fitness(error, self.features)

class Population:
    def __init__(self, size, dimension, fitness, crossover_prob=0.65, mutation_prob=0.1, elitism=True, sel_frac=0.3):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.dim = dimension
        self.fitness = fitness
        self.size = size
        self.population = [Individual(dimension) for _ in range(size)]
        self.parents = []
        self.offspring = []
        self.best_individual = self.population[0].chromosome  # initializing
        self.best_error = self.population[0].error            # initializing
        self.elitism = elitism
        self.selection_fraction = sel_frac

    def evaluate(self, error):
        for ind in self.population:
            ind.evaluate(self.fitness, error)
            if ind.error < self.best_error:
                self.best_error = ind.error
                self.best_position = ind.position

    def selection(self):
        # Tournament selection
        num_parents = int(self.selection_fraction * self.size)        
        # self.parents = sorted(self.population, key=lambda x: x.error)[:num_parents]
        parents = []      
        for _ in range(num_parents):
            sample = list(np.random.choice(self.population, 2, replace=False))
            if sample[0].error < sample[1].error:
                parents.append(sample[0])
            else:
                parents.append(sample[1])

        self.parents = parents
    
    def crossover(self):
        
        if self.elitism:
            num_offspring = self.size - len(self.parents)
        else:
            num_offspring = self.size
        
        new_offspring = []

        for _ in range(num_offspring // 2):
            prob = np.random.rand()
            sample = list(np.random.choice(self.parents, 2, replace=False))
            if prob <= self.crossover_prob:
                ind1 = sample[0].cross(sample[1])
                ind2 = sample[1].cross(sample[0])
            else:
                ind1 = sample[0]
                ind2 = sample[1]
            
            new_offspring.append(ind1)
            new_offspring.append(ind2)

        if self.elitism:
            self.offspring = self.parents + new_offspring
        else:    
            self.offspring = new_offspring
        
    def mutation(self):
        for ind in self.offspring:
            if np.random.rand() < self.mutation_prob:
                ind.mutate()

    def update_population(self):
        self.population = self.offspring

    def run_generation(self):
        self.selection()
        self.crossover()
        self.mutation()
        self.update_population()

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

def BinaryGA(max_epochs, N, fitness, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    dimension = fitness.dim()

    cross_prob = 0.65
    mut_prob = 0.05

    # create population
    population = Population(N, dimension, fitness, crossover_prob=cross_prob, mutation_prob=mut_prob)
    
        
    for epoch in range(max_epochs):

        population.run_generation()
    
        if epoch % 10 == 0 and epoch > 1:
            size = population.size
            best_error = population.best_error
            print("Generation = " + str(epoch) + " best error = %.3f" % best_error)
            print_position(population.best_position)
            print("Size: {}".format(size))

    print("")

    return population.best_position
