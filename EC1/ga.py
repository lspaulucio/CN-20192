# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 17/09/19
# Natural Computing - 2019/2
# Computacional Exercise 1

# Genetic Algorithm implementation

import numpy as np
import functions as F


def print_position(position):
    print("Best position: ", end="")
    print("[ ", end="")
    for i in range(len(position)):
        print("%.4f" % position[i], end="") # 4 decimals
        print(" ", end="")
    print("]")


class Individual:
    def __init__(self, dim, fitness, minx=None, maxx=None, position=None):
        self.dim = dim
        
        if position is None:
            if minx is None:
                minx = -10
            if maxx is None:
                maxx = 10
            self.position = (maxx - minx) * np.random.rand(dim) + minx
        else:
            minx, maxx = fitness.search_space()
            position[position < minx] = minx
            position[position > maxx] = maxx
            self.position = position

        self.fitness = fitness
        self.error = fitness(self.position)  # curr error

    def mutate(self, mi=0, sigma=1):
        beta = np.random.normal(mi, sigma)
        self.position *= beta
        self.error = self.fitness(self.position)

    def cross(self, other, beta):
        new_position = (beta * self.position) + (1 - beta) * other.position
        new = Individual(self.dim, self.fitness, position=new_position)
        return new

    def evaluate(self):
        self.error = self.fitness(self.position)


class Population:
    def __init__(self, size, dimension, fitness):
        minx, maxx = fitness.search_space()
        self.dim = dimension
        self.size = size
        self.population = [Individual(dimension, fitness, minx, maxx) for _ in range(size)]
        self.population.sort(key=lambda x: x.error)
        best_ind = self.population[0]
        self.best_position = best_ind.position
        self.best_error = best_ind.error

    def evaluate(self):
        best_ind = sorted(self.population, key=lambda x: x.error)[0]
        if best_ind.error < self.best_error:
            self.best_error = best_ind.error
            self.best_position = best_ind.position

    def selection(self, k=2):
        
        # best_individuals = []
        # for _ in range(self.size):
        #     sample = list(np.random.choice(self.population, k, replace=False))
        #     sample.sort(key=lambda x: x.error)
        #     best_individuals.append(sample[0])

        self.population.sort(key=lambda x: x.error)

    
    def crossover(self, cross_prob):
        parents_prob = 1 - cross_prob
        num_parents = int(parents_prob * self.size)
        parents = self.population[:num_parents]

        N = self.size - num_parents
        new_individuals = []

        for _ in range(N):
            sample = list(np.random.choice(parents, 2, replace=False))
            beta = np.random.rand()
            ind1 = sample[0].cross(sample[1], beta)
            ind2 = sample[1].cross(sample[0], beta)
                
            if ind1.error < ind2.error:
                new_individuals.append(ind1)
            else:
                new_individuals.append(ind2)

        self.population = parents + new_individuals
        
    def mutation(self, mut_prob, mi=0, sigma=1):
        for ind in self.population:
            prob = np.random.rand()
            if prob <= mut_prob:
                ind.mutate(mi, sigma)


def GA(max_epochs, N, fitness, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    dimension = fitness.dim()

    # create population
    population = Population(N, dimension, fitness)
    
    cross_prob = 0.65
    mut_prob = 0.1
    
    epoch = 0
    
    while epoch < max_epochs:

        population.selection()
        population.crossover(cross_prob)
        population.mutation(mut_prob)
        population.evaluate()
    
        if epoch % 10 == 0 and epoch > 1:
            size = population.size
            best_error = population.best_error
            print("Generation = " + str(epoch) + " best error = %.3f" % best_error)
            print_position(population.best_position)
            print("Size: {}".format(size))

        epoch += 1
    # while
    print("")

    return population.best_position


if __name__ == "__main__":

    fitness = F.Schwefel(dim=2)
    epochs = 1000

    best_position = GA(epochs, 20, fitness)

    print("Solution found after {} epochs".format(epochs))
    print_position(best_position)
    print("Best error: {:.3f}".format(fitness(best_position)))