# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:11:10 2024

@author: dogbl
"""
import numpy as np
import matplotlib.pyplot as plt

def absolute_difference(north, south):
    south = np.fliplr(np.rot90(south, k = 2))

    a = 0
    b = 20
    c = 8
    d = 73
    n_f = np.nanmedian(north[a:b,c:d])
    s_f = np.nanmedian(south[a:b,c:d])

    difference = north- (n_f/s_f)*south
    difference = np.clip(difference, 0, np.inf)

    
    return difference

