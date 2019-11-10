# -*- coding: utf-8 -*-

"""
Aluno: Leonardo Santos Paulucio
Data: 15/11/19
Computação Natural - Trabalho Computacional 2
"""

import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class MembershipFunction():

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

SHORT, MIDDLE, LONG = 0, 1, 2

class IrisFuzzyClassifier():
    

    def __init__(self, w1, w2, w3, w4):
        universe = np.linspace(0, 1, num=100)
        self.sepal_length = MembershipFunction(universe, w1)
        self.sepal_width = MembershipFunction(universe, w2)
        self.petal_length = MembershipFunction(universe, w3)
        self.petal_width = MembershipFunction(universe, w4)
        self.features = [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]

    def predict(self, data):
        y_pred = []

        for inp in data:
            x1 = self.sepal_length.activate(inp[0])
            x2 = self.sepal_width.activate(inp[1])
            x3 = self.petal_length.activate(inp[2])
            x4 = self.petal_width.activate(inp[3])
            y_pred.append(self.evaluate(x1, x2, x3, x4))

        return y_pred

    def score(self, x_test, y_true):
        y_pred = self.predict(x_test)
        return (y_true == y_pred).sum()/len(y_true)

    def evaluate(self, x1, x2, x3, x4):
        # R1 : IF x1_short or x1_ong and 
        #         x2_middle or x2_long and 
        #         x3_middle or x3_long and 
        #         x4_middle
        #     THEN iris=versicolor
        r11 = np.max([x1[SHORT], x1[LONG]])
        r12 = np.max([x2[MIDDLE], x2[LONG]])
        r13 = np.max([x3[MIDDLE], x3[LONG]])
        r14 = x4[MIDDLE]
        r1 = np.min([r11, r12, r13, r14])

        # R2 : IF x3_short or x3_middle and
        #         x4_short
        #     THEN iris=setosa

        r21 = np.max([x3[SHORT], x3[MIDDLE]])
        r22 = x4[SHORT]
        r2 = np.min([r21, r22])

        # R3 : IF x2_short or x2_middle and 
        #         x3_long and x4_long
            # THEN iris=verginica

        r31 = np.max([x2[SHORT], x2[MIDDLE]])
        r32 = x3[LONG]
        r3 = np.min([r31, r32])

        # R4 : IF x1_middle and 
        #         x2_short x2_middle and 
        #         x3_short and x4_long
        #     THEN iris=versicolor

        r41 = np.max(x1[MIDDLE])
        r42 = np.max([x2[SHORT], x2[MIDDLE]])
        r43 = x3[SHORT]
        r44 = x4[LONG]
        r4 = np.min([r41, r42, r43, r44])

        # Now we take our rules and apply them. 

        setosa = r2
        versicolor = np.max([r1, r4])
        virginica = r3

        return np.argmax([setosa, versicolor, virginica])