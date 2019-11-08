import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# Auto-membership function population is possible with .automf(3, 5, or 7)
quality.automf(3)
service.automf(3)
# 
# # Custom membership functions can be built interactively with a familiar,
# # Pythonic API
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8
# Crunch the numbers
tipping.compute()
print(tipping.output['tip'])



######################

# -*- coding: utf-8 -*-
"""
Aluno: Leonardo Santos Paulucio
Data: 15/11/19
Computação Natural - Trabalho Computacional 2

"""

import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def normalize(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

def make_membership_func(w=5):
    x = np.linspace(0, 1, num=100)
    lo = fuzz.trimf(x, [0, 0, w])
    md = fuzz.trimf(x, [0, w, 1])
    hi = fuzz.trimf(x, [w, 1, 1])

    return lo, md, hi

# Defining indices
# Species
SETOSA = 0
VERSICOLOR = 1
VIRGINICA = 2
SPECIES = [SETOSA, VERSICOLOR, VIRGINICA]

# Features
SEPAL_LENGTH = 0
SEPAL_WIDTH = 1
PETAL_LENGTH = 2
PETAL_WIDTH = 3
FEATURES = [SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]

# Importing iris dataset
iris = load_iris()


sizes = ['short', 'middle', 'long']
x1 = ctrl.Antecedent(np.arange(0,11,1), 'sepal_length') # sepal_length
x1.automf(names=sizes)
x2 = ctrl.Antecedent(np.arange(0,11,1), 'sepal_width') # sepal_width
x2.automf(names=sizes)
x3 = ctrl.Antecedent(np.arange(0,11,1), 'petal_length') # petal_length
x3.automf(names=sizes)
x4 = ctrl.Antecedent(np.arange(0,11,1), 'petal_width') # petal_width
x4.automf(names=sizes)

names = ['setosa', 'versicolor', 'virginica']
specie = ctrl.Consequent(np.arange(0, 3, 1), 'specie')
specie.automf(names=names)

r1 = ctrl.Rule((x1['short'] | x1['long']) &
               (x2['middle'] | x2['long']) &
               (x3['middle'] | x3['long']) &
               (x4['middle']), 
               specie['versicolor'])

r2 = ctrl.Rule((x3['short'] | x3['middle']) & 
               (x4['short']),
               specie['setosa'])

r3 = ctrl.Rule((x2['short'] | x2['middle']) &
               (x3['long']), 
               specie['virginica'])

r4 = ctrl.Rule((x1['middle']) & 
               (x2['short'] | x2['middle']) &
               (x3['short']) &
               (x4['long']), 
               specie['versicolor'])


system = ctrl.ControlSystem(rules=[r1, r2, r3, r4])

sim = ctrl.ControlSystemSimulation(system)

input = np.array([7.0,3.2,4.7,1.4]) #versicolor
input = np.array([6.5,3.0,5.2,2.0]) #virginica
input = np.array([5.1,3.5,1.4,0.2]) #setosa

sim.input['sepal_length'] = input[0]
sim.input['sepal_width'] = input[1]
sim.input['petal_length'] = input[2]
sim.input['petal_width'] = input[3]
sim.compute()
z = sim.output['specie']

print(z)
