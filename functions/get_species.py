# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:03:23 2024

@author: JDawg
"""
import numpy as np

def get_data_info(radiance, one_pixel, lower_bound, upper_bound, lower_bound_2, upper_bound_2, multi_regions = False):
    
    radiances_of_interest = np.array([])  # Ensure this is defined outside the if statement

    
    if multi_regions:
        lb1 = np.abs(one_pixel.data - lower_bound).argmin()
        lb2 = np.abs(one_pixel.data - lower_bound_2).argmin()
        ub1 = np.abs(one_pixel.data - upper_bound).argmin()
        ub2 = np.abs(one_pixel.data - upper_bound_2).argmin()

        radiances_of_interest = np.concatenate((radiance[:,:, lb1 : ub1 + 1 ], radiance[:,:, lb2 : ub2 + 1 ]), axis = -1 )
        
    brightnesses = 0.04*np.sum(radiances_of_interest, axis = -1)

    return brightnesses
                




            