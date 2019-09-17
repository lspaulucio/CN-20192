# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 17/09/19
# Natural Computing - 2019/2
# Computacional Exercise 1

# Genetic Algorithm implementation

import numpy as np
import functions as F
from utils import print_position


class Individual:
    def __init__(self, dim, minx=-10, maxx=10, position=None):
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        
        if position is None:
            self.position = (maxx - minx) * np.random.rand(dim) + minx
        else:
            minx, maxx = fitness.search_space()
            position[position < minx] = minx
            position[position > maxx] = maxx
            self.position = position

        self.error = float('inf')  # initial error

    def mutate(self, sigma):
        for i in range(self.dim):
            gene = self.position[i]
            beta = np.random.normal(gene, sigma)
            new_position = beta * self.position
            new_position[new_position < self.minx] = self.minx
            new_position[new_position > self.maxx] = self.maxx
        
        self.position = new_position

    def cross(self, other, beta):
        new_position = (beta * self.position) + (1 - beta) * other.position
        new = Individual(self.dim, self.minx, self.maxx, position=new_position)
        return new

    def evaluate(self, fitness):
        self.error = fitness(self.position)


class Population:
    def __init__(self, size, dimension, fitness, crossover_prob=0.65, mutation_prob=0.1):
        minx, maxx = fitness.search_space()
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.dim = dimension
        self.fitness = fitness
        self.size = size
        self.population = [Individual(dimension, minx, maxx) for _ in range(size)]
        self.parents = []
        self.offspring = []
        self.best_position = self.population[0].position
        self.best_error = self.population[0].error
        
        self.evaluate()

    def evaluate(self):
        for ind in self.population:
            ind.evaluate(self.fitness)
            if ind.error < self.best_error:
                self.best_error = ind.error
                self.best_position = ind.position

    def selection(self, selection_fraction=0.3):

        num_parents = int(selection_fraction * self.size)        
        # self.parents = sorted(self.population, key=lambda x: x.error)[:num_parents]

        parents = []
        
        for i in range(num_parents):
            sample = list(np.random.choice(self.population, 2, replace=False))
            if sample[0].error < sample[1].error:
                parents.append(sample[0])
            else:
                parents.append(sample[1])

        self.parents = parents

    
    def crossover(self):

        # num_offspring = self.size - len(self.parents)
        new_offspring = []

        for _ in range(self.size):
        # for _ in range(num_offspring):
            prob = np.random.rand()
            sample = list(np.random.choice(self.parents, 2, replace=False))
            if prob <= self.crossover_prob:
                beta = np.random.rand()
                ind1 = sample[0].cross(sample[1], beta)
                ind1.evaluate(self.fitness)
                ind2 = sample[1].cross(sample[0], beta)
                ind2.evaluate(self.fitness)
            else:
                ind1 = sample[0]
                ind2 = sample[1]
            
            if ind1.error < ind2.error:
                new_offspring.append(ind1)
            else:
                new_offspring.append(ind2)

        self.offspring = new_offspring
        
    def mutation(self, sigma=1):
        for ind in self.offspring:
            prob = np.random.rand()
            if prob <= self.mutation_prob:
                ind.mutate(sigma)

    def run_generation(self):
        self.selection()
        self.crossover()
        self.mutation(sigma=0.3)
        # self.population = self.parents + self.offspring
        self.population = self.offspring
        self.evaluate()

        

def GA(max_epochs, N, fitness, seed=None):
    
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


if __name__ == "__main__":

    fitness = F.Schwefel(dim=2)
    epochs = 500

    best_position = GA(epochs, 50, fitness)

    print("Solution found after {} epochs".format(epochs))
    print_position(best_position)
    print("Best error: {:.3f}".format(fitness(best_position)))