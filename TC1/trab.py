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
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
    
    # Training parameters
    BATCH_SIZE = 32
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    learning_rate = 1e-3

    model = Classifier(NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES)

    fitness_func = Fitness(NUM_FEATURES, ro=RO)
    
    population = Population(size=POPULATION_SIZE, 
                            dimension=NUM_FEATURES, 
                            fitness=fitness_func, 
                            crossover_prob=CROSSOVER_PROB, 
                            mutation_prob=MUTATION_PROB,
                            elitism=True,
                            sel_frac=0.3)

    if cuda:
        print("Cuda available!!! Model will run on GPU")
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        # model.train()
        # running_loss = 0.0

        # for x, y in train_loader(dataset):
        # get the inputs; data is a list of [inputs, labels]
        # sentence, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        # optimizer.zero_grad()
        
        # for ind in population:
        #    features_mask = torch.tensor(ind.chromosome)
        #    features = x * features_mask

        # outputs = model(features)
        # loss = criterion(outputs, y)
        # loss.backward()
        # optimizer.step()

        # check shapes for feature selection
        # multiply 
        # get loss and evaluate individuals

        # print statistics
        # running_loss += loss.item()
        # if i % 49 == 0:    # print every 500 mini-batches
        #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 49))
        #     running_loss = 0.0
        
        population.run_generation()


# correct = 0
# total = 0
# with torch.no_grad():
#     model.eval()
#     for data in tqdm(test_loader):
#         sentence, labels = data[0].to(device), data[1].to(device)
#         outputs = model(sentence.cuda())
#         # import pdb; pdb.set_trace()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the %d test images: %.3f %%' % (total, 100 * correct / total))