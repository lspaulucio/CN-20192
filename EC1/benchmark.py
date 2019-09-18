# -*- coding: utf-8 -*-

from functions import Ackley, Rastrigin, Rosenbrock, Griewank, Schwefel, Langermann

from pso import PSO
from bbpso import BBPSO
from ga import GA
from cma import CMA

import csv
import json
import numpy as np

function_names = ["Ackley", "Rastrigin", "Rosenbrock", "Griewank", "Schwefel"]
algorithms_names = ['PSO', 'BBPSO', 'GA', 'CMA-ES']

def run_benchmark(num_tests, population_size, epochs, D):
    print("Starting Benchmark...")
    
    functions = [Ackley(D), Rastrigin(D), Rosenbrock(D), Griewank(D), Schwefel(D)]
    results = {f:{a:[] for a in algorithms_names} for f in function_names}
    for fitness in functions:
        for i in range(num_tests):
            func_name = fitness.__class__.__name__
            results[func_name]['PSO'].append(fitness(PSO(epochs, population_size, fitness)))
            results[func_name]['BBPSO'].append(fitness(BBPSO(epochs, population_size, fitness)))
            results[func_name]['GA'].append(fitness(GA(epochs, population_size, fitness)))
            results[func_name]['CMA-ES'].append(fitness(CMA(epochs, population_size, fitness)))

    return results


def save_results_json(results, filename):
    with open(filename, 'w') as json_file:
            json.dump(results, json_file)

def compile_results(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        for f in function_names:
            for a in algorithms_names:
                r = np.array(data[f][a])
                best = r.min()
                worst = r.max()
                mean = r.mean()
                std = r.std()
                data[f][a] = [best, worst, mean, std]

    return data

def save_csv(results, filename):
    with open(filename, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Function', 'Algorithm', 'Best', 'Worst', 'Mean', 'Std'])
        for f in function_names:
            for a in algorithms_names:
                r = np.round(results[f][a], 4)
                filewriter.writerow([f, a, *r])

if __name__ == "__main__":
    
    # num_tests = 50
    # population_size = 100
    # epochs = 100
    # dimensions = [2, 3, 10, 20, 30]
    # 
    # for dim in dimensions:
        # results = run_benchmark(num_tests, population_size, epochs, dim)
        # save_results_json(results, filename='results/results_{}.json'.format(dim))    
        # compiled = compile_results('results/results_{}.json'.format(dim))
        # save_csv(compiled, 'results/compiled_results_{}.csv'.format(dim))

    compiled = compile_results('results/results_2.json')
    save_csv(compiled, 'results/compiled_results_2.csv')

   


    
    