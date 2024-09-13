# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:11:10 2024

@author: dogbl
"""
import numpy as np

def absolute_difference(north, south):
    
    difference = north - np.fliplr(np.rot90(south, k = 2))
    difference = np.abs(difference)
    
    return difference