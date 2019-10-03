# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 15/10/19
# Natural Computing - 2019/2
# Computacional Assignment 1

"""     - - - - - - -     Selected Features     - - - - - - -  
        -           -     - - - - - - - - >     -           -  
        - Binary GA -                           -    NN     -  
        -           -     < - - - - - - - -     -           -  
        - - - - - - -          Loss             - - - - - - -
""" 

import numpy as np
import random
import torch

from utils import Fitness, print_position
from ga import Population
from classifier import Classifier

if __name__ == "__main__":

    # Setting seed for reproducibility
    SEED = 500
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Hyperparameters
    NUM_EPOCHS = 50
    NUM_FEATURES = 10
    HIDDEN_SIZE = 20
    NUM_CLASSES = 5
    POPULATION_SIZE = 10
    MUTATION_PROB = 0.1
    CROSSOVER_PROB = 0.9
    RO = 0.9
    
    net = Classifier(NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES)
    fitness_func = Fitness(NUM_FEATURES, ro=RO)
    
    population = Population(size=POPULATION_SIZE, 
                            dimension=NUM_FEATURES, 
                            fitness=fitness_func, 
                            crossover_prob=CROSSOVER_PROB, 
                            mutation_prob=MUTATION_PROB)

    for epoch in range(NUM_EPOCHS):

        population.run_generation()
    
        if epoch % 10 == 0 and epoch > 1:
            size = population.size
            best_error = population.best_error
            print("Generation = " + str(epoch) + " best error = %.3f" % best_error)
            print_position(population.best_position)
            print("Size: {}".format(size))

    print("")
