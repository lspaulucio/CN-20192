# -*- coding: utf-8 -*-
# Aluno: Leonardo Santos Paulucio
# Data: 17/09/19
# Natural Computing - 2019/2
# Computacional Exercise 1

# PSO implementation

# Based on James D. McCaffrey implementation available on:
# https://jamesmccaffrey.wordpress.com/2015/06/09/particle-swarm-optimization-using-python/

import sys
import copy
import random
import numpy as np
import functions as F


def print_position(position):
    print("Best position: ", end="")
    print("[ ", end="")
    for i in range(len(position)):
        print("%.4f" % position[i], end="") # 4 decimals
        print(" ", end="")
    print("]")


class Particle:
    def __init__(self, dim, minx, maxx, fitness):
        self.position = (maxx - minx) * np.random.rand(dim) + minx
        self.velocity = (maxx - minx) * np.random.rand(dim) + minx
        self.error = fitness(self.position)  # curr error
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.pbest_err = self.error  # best error

    def move(self):
        self.position += self.velocity


def PSO(max_epochs, n, fitness, seed=None):
    
    if seed is not None:
        rnd = random.Random(seed)
    else:
        rnd = random.Random()


    dim = fitness.dim()
    minx, maxx = fitness.search_space()

    # create n random particles
    swarm = [Particle(dim, minx, maxx, fitness) for i in range(n)]

    best_swarm_pos = [0.0 for i in range(dim)]  # not necess.
    best_swarm_err = sys.float_info.max  # swarm best
    
    for i in range(n):  # check each particle
        if swarm[i].error < best_swarm_err:
            best_swarm_err = swarm[i].error
            best_swarm_pos = copy.copy(swarm[i].position)

    epoch = 0
    chi = 0.7298    # inertia
    c1 = 2.05       # cognitive (particle)
    c2 = 2.05       # social (swarm)

    while epoch < max_epochs:

        if epoch % 10 == 0 and epoch > 1:
            print("Epoch = " + str(epoch) + " best error = %.3f" % best_swarm_err)

        for i in range(n):  # process each particle
            # compute new velocity of curr particle
            for k in range(dim):
                r1 = rnd.random()    # randomizations
                r2 = rnd.random()

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
            swarm[i].error = fitness(swarm[i].position)

            # is new position a new best for the particle?
            if swarm[i].error < swarm[i].pbest_err:
                swarm[i].pbest_err = swarm[i].error
                swarm[i].pbest_position = copy.copy(swarm[i].position)

            # is new position a new best overall?
            if swarm[i].error < best_swarm_err:
                best_swarm_err = swarm[i].error
                best_swarm_pos = copy.copy(swarm[i].position)

        # for-each particle
        epoch += 1
    # while
    print("")
    return best_swarm_pos
# end Solve


if __name__ == "__main__":

    fitness = F.Ackley(dim=3)
    
    epochs = 100

    best_position = PSO(epochs, 100, fitness)

    print("Solution found after {} epochs".format(epochs))
    print_position(best_position)
    print("Best error: {:.3f}".format(fitness(best_position)))
