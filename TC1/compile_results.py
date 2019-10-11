# -*- coding: utf-8 -*-

import os, glob
import pickle
import numpy as np

def get_results(data):

    individuals = data['individual']

    scores = np.array([i.score for i in individuals])
    scores_info = [scores.mean(), scores.min(), scores.max(), scores.std()]
    print('Scores: ', scores_info)

    fitness = np.array([i.error for i in individuals])
    fitness_info = [fitness.mean(), fitness.min(), fitness.max(), fitness.std()]
    print('Fitness: ', fitness_info)

    num_features = np.array([i.features for i in individuals])
    features_info = [num_features.mean(), num_features.min(), num_features.max(), num_features.std()]
    print('Num. Features: ', features_info)

    time = np.array(data['time'])
    time_info = [time.mean(), time.min(), time.max(), time.std()]
    print('Time: ', time_info, '\n')

if __name__ == "__main__":
    
    for f in glob.glob('results/*.pickle'):    
        file = open(f, 'rb')
        print('Compiling results of file: "%s"' % f)
        data = pickle.load(file)
        get_results(data)