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
import pickle
import warnings
import argparse
import numpy as np

from ga import Population
from classifiers import NeuralClassifier, ELM
from utils import Fitness
from datasets import Arrhythmia, Ionosphere, Wine

from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Trabalho 3')
    parser.add_argument('--dataset', action='store', dest='dataset', required=True, choices=['wine', 'ionosphere', 'arrhythmia'], help='Dataset que sera usado')

    args = parser.parse_args()

    # Setting seed for reproducibility
    SEED = 500
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Parameters
    NUM_RUNS = 20
    NUM_GENERATIONS = 50
    NUM_EPOCHS = 50
    
    # GA Parameters
    POPULATION_SIZE = 10
    MUTATION_PROB = 0.1
    CROSSOVER_PROB = 0.9
    RO = 0.9

    # Training parameters
    LEARNING_RATE = 1e-3

    # Features - Arrhythmia 279, Ionosphere 34, Wine 13
    # Classes - Arrhythmia 16, Ionosphere 2, Wine 3

    if args.dataset == 'wine':
        load_dataset = Wine
        NUM_FEATURES = 13
        NUM_CLASSES = 3
    elif args.dataset == 'ionosphere':
        load_dataset = Ionosphere
        NUM_FEATURES = 34
        NUM_CLASSES = 2
    elif args.dataset == 'arrhythmia':
        load_dataset = Arrhythmia
        NUM_FEATURES = 279
        NUM_CLASSES = 16

    # Loading dataset
    x_train, y_train, x_test, y_test = load_dataset()
    
    HIDDEN_SIZE = int(np.sqrt(NUM_FEATURES * NUM_CLASSES))
    
    fitness_func = Fitness(NUM_FEATURES, ro=RO)
    
    individuals = []
    times = []

    for step in range(NUM_RUNS):
        
        start_time = time.time()

        population = Population(size=POPULATION_SIZE, 
                            dimension=NUM_FEATURES, 
                            fitness=fitness_func, 
                            crossover_prob=CROSSOVER_PROB, 
                            mutation_prob=MUTATION_PROB,
                            sel_frac=0.2)

        for generation in range(NUM_GENERATIONS):
            
            for ind in population:
                feature_mask = ind.chromosome
                x_tr = feature_mask * x_train
                x_ts = feature_mask * x_test
                model = NeuralClassifier(HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS)
                # model = ELM(HIDDEN_SIZE)
                model.fit(x_tr, y_train)
                score = model.score(x_ts, y_test)
                error = 1 - score
                ind.evaluate(fitness_func, error)
                ind.score = score
            
            population.evaluate()
            population.new_generation()
            
            print("Running step: %d" % (step + 1))
            print("Generation number: %d" % (generation + 1))
            print("Best individue fitness: %.4f\nNumber of features selected: %d \nScore: %.3f" % (population.best_error, 
                                                                                                   population.best_individual.features, 
                                                                                                   population.best_individual.score))
            print("Features selected: {}\n".format([i for i in range(NUM_FEATURES) if population.best_individual.chromosome[i] == 1]))
        
        exec_time = time.time() - start_time

        individuals.append(population.best_individual)
        times.append(exec_time)
    
    info = {'individual': individuals,
            'time': times}
    
    with open("results/NN_arrhythmia.pickle", "wb") as f:
        pickle.dump(info, f)