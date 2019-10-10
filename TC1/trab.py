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

import time
import random
import warnings
import numpy as np

from ga import Population
from classifiers import NeuralClassifier, ELM
from utils import Fitness, print_position
from datasets import Arrhythmia, Ionosphere, Wine

from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":

    # Setting seed for reproducibility
    SEED = 500
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Parameters
    NUM_RUNS = 1
    NUM_GENERATIONS = 50
    NUM_EPOCHS = 20
    
    # Features - Arrhythmia 279, Ionosphere 34, Wine 13
    NUM_FEATURES = 279
    # Classes - Arrhythmia 16, Ionosphere 2, Wine 3
    NUM_CLASSES = 16

    # Loading dataset
    x_train, y_train, x_test, y_test = Arrhythmia()
    
    HIDDEN_SIZE = int(np.sqrt(NUM_FEATURES * NUM_CLASSES))
    POPULATION_SIZE = 10
    MUTATION_PROB = 0.1
    CROSSOVER_PROB = 0.9
    RO = 0.9
    
    # Training parameters
    LEARNING_RATE = 1e-3

    fitness_func = Fitness(NUM_FEATURES, ro=RO)
    
    individuals = []
    times = []

    for step in range(NUM_RUNS):
        
        population = Population(size=POPULATION_SIZE, 
                            dimension=NUM_FEATURES, 
                            fitness=fitness_func, 
                            crossover_prob=CROSSOVER_PROB, 
                            mutation_prob=MUTATION_PROB,
                            sel_frac=0.2)

        for generation in range(NUM_GENERATIONS):

            start_time = time.time()
            
            for ind in population:
                feature_mask = ind.chromosome
                x_tr = feature_mask * x_train
                x_ts = feature_mask * x_test
                model = NeuralClassifier(HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS)
                model = ELM(HIDDEN_SIZE)
                model.fit(x_tr, y_train)
                score = model.score(x_ts, y_test)
                error = 1 - score
                ind.evaluate(fitness_func, error)
                ind.score = score
            
            population.evaluate()
            population.new_generation()
            
            print("Running step: %d" % (step + 1))
            print("Generation number: %d" % (generation + 1))
            print("Best individue fitness: %.4f\nNumber of features selected: %d \nScore: %.3f\n" % (population.best_error, 
                                                                                                     population.best_individual.features, 
                                                                                                     population.best_individual.score))
            
            end_time = time.time()
            exec_time = end_time - start_time
            individuals.append(population.best_individual)
            times.append(exec_time)