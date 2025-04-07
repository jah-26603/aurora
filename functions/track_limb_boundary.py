# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:01:02 2025

@author: JDawg
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import medfilt

def variance(data, smooth_line):
    n = len(data)
    return np.mean((data - smooth_line) ** 2) 

def track_limb_boundary(species_datastructure, species, low_cadence = False):
    
    limb_boundaries = {}
    
    for spec in species:
        aa = species_datastructure[spec]['points']

        if low_cadence is False: 
            bb = np.column_stack((medfilt(aa[:, 0], kernel_size = 9),
                                  medfilt(aa[:, 1], kernel_size = 9)))
            c, r = np.where(np.abs(aa - bb) > 3)
        else:
            bb = np.column_stack((medfilt(aa[:, 0], kernel_size = 3),
                                  medfilt(aa[:, 1], kernel_size = 3)))
            c, r = np.where(np.abs(aa - bb) > 4)
        # aa[c, r] = bb[c, r]
        
        limb_boundaries[spec] = {}
        limb_boundaries[spec]['pts'] = aa
        limb_boundaries[spec]['smth_pts'] = bb
        limb_boundaries[spec]['variance'] = variance(aa, bb)
        
        plt.figure()
        plt.plot(aa)
        plt.plot(bb)
        plt.title(spec)
        plt.show()
        
        
    best_spec = min(limb_boundaries, key=lambda x: np.sum(limb_boundaries[x]['variance']))
    boundary_choice = limb_boundaries[best_spec]['pts']

    return boundary_choice