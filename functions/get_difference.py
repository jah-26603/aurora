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
    b = 30
    c = 20
    d = 72
    n_f = np.nanmean(north[a:b,c:d])
    s_f = np.nanmean(south[a:b,c:d])
    n_f= 1  
    s_f = 1
    difference = north- (n_f/s_f)*south
    difference = np.abs(difference)
    difference = north - south
    difference = np.abs(difference)
    
    return difference

