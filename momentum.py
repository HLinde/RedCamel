# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

def make_momentum(number_of_momenta=1000):
    momentum = np.random.randn(number_of_momenta, 1, 3)
    return momentum

momentum = make_momentum(10)
print(momentum[:,0,0])

        