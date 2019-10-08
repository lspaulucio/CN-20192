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
from tqdm import tqdm

from utils import Fitness, print_position
from ga import Population

from sklearn.metrics import confusion_matrix, f1_score, recall_score
from classifier import Classifier

# cm=confusionmatrix
# sns.heatmap(cm, center=True)

if __name__ == "__main__":

    # Setting seed for reproducibility
    SEED = 500
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Hyperparameters
    NUM_EPOCHS = 50
    NUM_FEATURES = 10
    NUM_CLASSES = 5
    HIDDEN_SIZE = np.sqrt(NUM_FEATURES * NUM_CLASSES)
    POPULATION_SIZE = 10
    MUTATION_PROB = 0.1
    CROSSOVER_PROB = 0.9
    RO = 0.9
    
    # Training parameters
    LEARNING_RATE = 1e-3

    model = Classifier(HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS, SEED)
    
    fitness_func = Fitness(NUM_FEATURES, ro=RO)
    
    population = Population(size=POPULATION_SIZE, 
                            dimension=NUM_FEATURES, 
                            fitness=fitness_func, 
                            crossover_prob=CROSSOVER_PROB, 
                            mutation_prob=MUTATION_PROB,
                            elitism=True,
                            sel_frac=0.3)


