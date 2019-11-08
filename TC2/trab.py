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

class Membership():

    def __init__(self, universe, w):
        self.universe = universe
        self.w = w
        self.membership = list(self.make_membership_func(universe, w))

    def activate(self, input):
        self.activation = [fuzz.interp_membership(self.universe, i, input) for i in self.membership]
        return self.activation

    def make_membership_func(self, universe, w=0.5):
        lo = fuzz.trimf(universe, [0, 0, w])
        md = fuzz.trimf(universe, [0, w, 1])
        hi = fuzz.trimf(universe, [w, 1, 1])
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
SHORT, MIDDLE, LONG = 0, 1, 2
universe = np.linspace(0, 1, num=100)
w1 = 0.5
w2 = 0.5
w3 = 0.5
w4 = 0.5

# Generate fuzzy membership functions
# sepal_length_lo, sepal_length_md, sepal_length_hi = make_membership_func(universe, w1)
# sepal_width_lo, sepal_width_md, sepal_width_hi = make_membership_func(universe, w2)
# petal_length_lo, petal_length_md, petal_length_hi = make_membership_func(universe, w3)
# petal_width_lo, petal_width_md, petal_width_hi = make_membership_func(universe, w4)

sepal_length = Membership(universe, w1)
sepal_width = Membership(universe, w2)
petal_length = Membership(universe, w3)
petal_width = Membership(universe, w4)

# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
# qual_level_lo = fuzz.interp_membership(universe, qual_lo, 6.5)
# qual_level_md = fuzz.interp_membership(universe, qual_md, 6.5)
# qual_level_hi = fuzz.interp_membership(universe, qual_hi, 6.5)


# serv_level_lo = fuzz.interp_membership(universe, serv_lo, 9.8)
# serv_level_md = fuzz.interp_membership(universe, serv_md, 9.8)
# serv_level_hi = fuzz.interp_membership(universe, serv_hi, 9.8)

x1 = sepal_length.activate(inp)
x2 = sepal_width.activate(inp)
x3 = petal_length.activate(inp)
x4 = petal_width.activate(inp)

# r1 = ctrl.Rule((x1['short'] | x1['long']) &
#                (x2['middle'] | x2['long']) &
#                (x3['middle'] | x3['long']) &
#                (x4['middle']), 
#                specie['versicolor'])

r11 = np.max(x1[SHORT], x1[LONG])
r12 = np.max(x2[MIDDLE], x2[LONG])
r13 = np.max(x3[MIDDLE], x3[LONG])
r14 = x4[MIDDLE]
r1 = np.min([r11, r12, r13, r14])

# r2 = ctrl.Rule((x3['short'] | x3['middle']) & 
#                (x4['short']),
#                specie['setosa'])

r21 = np.max(x3[SHORT], x3[MIDDLE])
r22 = x4[SHORT]
r2 = np.min([r21, r22])

# r3 = ctrl.Rule((x2['short'] | x2['middle']) &
#                (x3['long']), 
#                specie['virginica'])

r31 = np.max(x2[SHORT], x2[MIDDLE])
r32 = x3[LONG]
r3 = np.min([r31, r32])

# r4 = ctrl.Rule((x1['middle']) & 
#                (x2['short'] | x2['middle']) &
#                (x3['short']) &
#                (x4['long']), 
#                specie['versicolor'])

r41 = np.max(x1[MIDDLE])
r42 = np.max(x2[SHORT], x2[MIDDLE])
r43 = x3[SHORT]
r44 = x4[LONG]
r4 = np.min([r41, r42, r43, r44])

# Now we take our rules and apply them. 
# 1. If the food is bad OR the service is poor, then the tip will be low
# 2. If the service is acceptable, then the tip will be medium
# 3. If the food is great OR the service is amazing, then the tip will be high

tip_lo = fuzz.trimf(x_tip, [0, 0, 13])
tip_md = fuzz.trimf(x_tip, [0, 13, 25])
tip_hi = fuzz.trimf(x_tip, [13, 25, 25])

# The OR operator means we take the maximum of these two.
active_rule1 = np.fmax(qual_level_lo, serv_level_lo)

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
tip_activation_lo = np.fmin(active_rule1, tip_lo) # removed entirely to 0

# For rule 2 we connect acceptable service to medium tipping
tip_activation_md = np.fmin(serv_level_md, tip_md)

# For rule 3 we connect high service OR high food with high tipping
active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
tip_activation_hi = np.fmin(active_rule3, tip_hi)
tip0 = np.zeros_like(x_tip)

# print(tip_activation_hi)#, tip_activation_md, tip_activation_hi)

aggregated = np.fmax(tip_activation_lo, np.fmax(tip_activation_md, tip_activation_hi))
print(np.max(tip_activation_lo))
print(np.max(tip_activation_md))
print(np.max(tip_activation_hi))
# print(aggregated)
# Calculate defuzzified result
tip = fuzz.defuzz(x_tip, aggregated, 'centroid')
tip_activation = fuzz.interp_membership(x_tip, aggregated, tip)

# print(tip)