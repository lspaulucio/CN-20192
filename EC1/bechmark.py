# -*- coding: utf-8 -*-

from functions import Ackley, Rastrigin, Rosenbrock, Griewank, Schwefel, Langermann

from pso import PSO
from bbpso import BBPSO
from ga import GA
from cma import CMA
import json

if __name__ == "__main__":

    print("Benchmark")

    num_tests = 30
    population_size = 50
    epochs = 100

    D = 2  # dimension

    
    functions = [Ackley(D), Rastrigin(D), Rosenbrock(D), Griewank(D), Schwefel(D), Langermann(D)]
    algorithms = ['PSO', 'BBPSO', 'GA', 'CMA-ES']
    results = {f.__class__.__name__:{a:[] for a in algorithms} for f in functions}

    for fitness in functions:
        for i in range(num_tests):
            alg_name = fitness.__class__.__name__
            results[alg_name]['PSO'].append(fitness(PSO(epochs, population_size, fitness)))
            results[alg_name]['BBPSO'].append(fitness(BBPSO(epochs, population_size, fitness)))
            results[alg_name]['GA'].append(fitness(GA(epochs, population_size, fitness)))
            results[alg_name]['CMA-ES'].append(fitness(CMA(epochs, population_size, fitness)))

    with open('results.json', 'w') as json_file:
        json.dump(results, json_file)


    
    