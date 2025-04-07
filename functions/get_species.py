# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:03:23 2024

@author: JDawg
"""
import numpy as np
import matplotlib.pyplot as plt

def get_data_info(radiance, one_pixel, lower_bound, upper_bound, lower_bound_2 = 0 , upper_bound_2 = 0, multi_regions = False):
    
    radiances_of_interest = np.array([])  # Ensure this is defined outside the if statement

    lb1 = np.abs(one_pixel.data - lower_bound).argmin()
    ub1 = np.abs(one_pixel.data - upper_bound).argmin()
    if multi_regions:
        lb2 = np.abs(one_pixel.data - lower_bound_2).argmin()
        ub2 = np.abs(one_pixel.data - upper_bound_2).argmin()
        radiances_of_interest = np.concatenate((radiance[:,:, lb1 : ub1 + 1 ], radiance[:,:, lb2 : ub2 + 1 ]), axis = -1 )
    else:
        radiances_of_interest = radiance[:,:, lb1 : ub1 + 1 ]
    brightnesses = 0.04*np.nansum(radiances_of_interest, axis = -1)
    
    condition = np.sum(radiance == 0, axis=2) < 5  # Checks for stars along emission profile of each pixel, might need a different check
    brightnesses[condition] = 0  # Set brightnesses to 0 where condition is True
    

    return brightnesses
                




            