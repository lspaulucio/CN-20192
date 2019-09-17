# -*- coding: utf-8 -*-

def print_position(position):
    print("Best position: ", end="")
    print("[ ", end="")
    for i in range(len(position)):
        print("%.4f" % position[i], end="") # 4 decimals
        print(" ", end="")
    print("]")