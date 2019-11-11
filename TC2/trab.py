# -*- coding: utf-8 -*-
"""
Aluno: Leonardo Santos Paulucio
Data: 15/11/19
Computação Natural - Trabalho Computacional 2

"""

import time
import random
import pickle
import numpy as np
from copy import deepcopy
from pso import Particle
from ga import Population
from sklearn.datasets import load_iris
from fuzzy_classifier import IrisFuzzyClassifier

def normalize(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

def GA():
    # Parameters
    NUM_RUNS = 20
    NUM_GENERATIONS = 50
    NUM_FEATURES = 4
    
    # GA Parameters
    POPULATION_SIZE = 20
    MUTATION_PROB = 0.1
    CROSSOVER_PROB = 0.9

    # Importing iris dataset
    iris = load_iris()
    data = normalize(iris.data)
    y_true = iris.target
    
    individuals = []
    times = []

    for step in range(NUM_RUNS):
        
        start_time = time.time()

        population = Population(size=POPULATION_SIZE, 
                                dimension=NUM_FEATURES, 
                                min=0, max=1,
                                crossover_prob=CROSSOVER_PROB, 
                                mutation_prob=MUTATION_PROB,
                                sel_frac=0.2)

        for generation in range(NUM_GENERATIONS):
            
            for ind in population:
                model = IrisFuzzyClassifier(*ind.position)
                score = model.score(data, y_true)
                error = 1 - score
                ind.error = error
                ind.score = score
            
            population.evaluate()
            population.new_generation()
            
            print("Running step: %d" % (step + 1))
            print("Generation number: %d" % (generation + 1))
            print("Best individue: {}".format(population.best_individual.position))
            print("Best individue fitness: %.4f\nScore: %.3f" % (population.best_error, 
                                                                 1 - population.best_error))
            print(len(population.population))
        
        exec_time = time.time() - start_time

        individuals.append(population.best_individual)
        times.append(exec_time)
    
    info = {'individual': individuals,
            'time': times}
    
    with open("results/iris_fuzzy_ga.pickle", "wb") as f:
        pickle.dump(info, f)

def PSO():
    # Parameters
    NUM_RUNS = 20
    NUM_GENERATIONS = 50
    NUM_FEATURES = 4
    
    POPULATION_SIZE = 20

    minx, maxx = 0.0, 1.0

    iris = load_iris()
    data = normalize(iris.data)
    y_true = iris.target


    epoch = 0
    chi = 0.7298    # inertia
    c1 = 2.05       # cognitive (particle)
    c2 = 2.05       # social (swarm)

    positions = []
    errors = []
    times = []
        
    for step in range(NUM_RUNS):
        start_time = time.time()
        
        # create n random particles
        swarm = [Particle(NUM_FEATURES, minx, maxx) for i in range(POPULATION_SIZE)]

        best_swarm_pos = [0.0 for i in range(NUM_FEATURES)]  # not necess.
        best_swarm_err = float('inf')  # swarm best
        
        for epoch in range(NUM_GENERATIONS):

            for i in range(POPULATION_SIZE):  # process each particle
                # compute new velocity of curr particle
                for k in range(NUM_FEATURES):
                    r1 = np.random.random()    # randomizations
                    r2 = np.random.random()

                    swarm[i].velocity[k] =  chi * (swarm[i].velocity[k] +
                                            (c1 * r1 * (swarm[i].pbest_position[k] - swarm[i].position[k])) + # personal
                                            (c2 * r2 * (best_swarm_pos[k] - swarm[i].position[k])))            # global

                    # compute new position using new velocity
                    new_position = swarm[i].position[k] + swarm[i].velocity[k]
                    
                    if new_position < minx:
                        new_position = minx
                    elif new_position > maxx:
                        new_position = maxx
                    
                    swarm[i].position[k] = new_position

                # compute error of new position
                position = swarm[i].position
                model = IrisFuzzyClassifier(*position)
                score = model.score(data, y_true)
                swarm[i].error = 1 - score

                # is new position a new best for the particle?
                if swarm[i].error < swarm[i].pbest_err:
                    swarm[i].pbest_err = swarm[i].error
                    swarm[i].pbest_position = deepcopy(swarm[i].position)

                # is new position a new best overall?
                if swarm[i].error < best_swarm_err:
                    best_swarm_err = swarm[i].error
                    best_swarm_pos = deepcopy(swarm[i].position)

                print("Running step: %d" % (step + 1))
                print("Generation number: %d" % (epoch + 1))
                print("Best individue: {}".format(best_swarm_pos))
                print("Best individue fitness: %.4f\nScore: %.3f" % (best_swarm_err, 
                                                                    1 - best_swarm_err))

        exec_time = time.time() - start_time
        positions.append(best_swarm_pos)
        errors.append(best_swarm_err)
        times.append(exec_time)

    info = {'positions': positions,
            'errors': errors,
            'time': times}
    
    with open("results/iris_fuzzy_pso.pickle", "wb") as f:
        pickle.dump(info, f)

if __name__ == "__main__":

    # Setting seed for reproducibility
    SEED = 500
    random.seed(SEED)
    np.random.seed(SEED)

    # GA()
    PSO()
    