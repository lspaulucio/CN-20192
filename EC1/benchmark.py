# -*- coding: utf-8 -*-

from functions import Ackley, Rastrigin, Rosenbrock, Griewank, Schwefel, Langermann

from pso import PSO
from bbpso import BBPSO
from ga import GA
from cma import CMA
import json
import numpy as np

function_names = ["Ackley", "Rastrigin", "Rosenbrock", "Griewank", "Schwefel", "Langermann"]
algorithms_names = ['PSO', 'BBPSO', 'GA', 'CMA-ES']

def run_benchmark(num_tests, population_size, epochs, D):
    print("Starting Benchmark...")
    
    functions = [Ackley(D), Rastrigin(D), Rosenbrock(D), Griewank(D), Schwefel(D), Langermann(D)]
    results = {f:{a:[] for a in algorithms_names} for f in function_names}
    print(results)
    exit()
    for fitness in functions:
        for i in range(num_tests):
            func_name = function_names[i]
            results[func_name]['PSO'].append(fitness(PSO(epochs, population_size, fitness)))
            results[func_name]['BBPSO'].append(fitness(BBPSO(epochs, population_size, fitness)))
            results[func_name]['GA'].append(fitness(GA(epochs, population_size, fitness)))
            results[func_name]['CMA-ES'].append(fitness(CMA(epochs, population_size, fitness)))

    save_results_json(results, filename='results.json')    


def save_results_json(results, filename):
    with open(filename, 'w') as json_file:
            json.dump(results, json_file)

def compile_results(results, filename):
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
    
    

if __name__ == "__main__":
    
    num_tests = 30
    population_size = 50
    epochs = 100
    dim = 2  # dimension

    run_benchmark(num_tests, population_size, epochs, dim)

   


    
    