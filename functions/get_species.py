# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:03:23 2024

@author: JDawg
"""
import numpy as np
import math

def get_data_info(radiance, radiance_unc, one_pixel, lower_bound, upper_bound, lower_bound_2, upper_bound_2, multi_regions = False, all_data = True):
    
    indices_of_interest = np.where(np.logical_and(one_pixel >= lower_bound, one_pixel <= upper_bound))[0]  
    
    #removes a region of wavelengths from consideration
    if multi_regions:
        indices_of_interest= indices_of_interest[~np.logical_and(one_pixel[indices_of_interest] > lower_bound_2
                                                               , one_pixel[indices_of_interest] < upper_bound_2)]
  
        
        
    
    radiances_of_interest = np.asarray([[[np.NaN]*len(indices_of_interest)]*92]*104)
    radiances_of_interest_unc = np.asarray([[[np.NaN]*len(indices_of_interest)]*92]*104)

    brightnesses = np.asarray([[np.NaN]*92]*104)
    brightnesses_unc = np.asarray([[np.NaN]*92]*104)

    
    for i in range(104):
        for j in range(92):
            for k1, k2 in enumerate(indices_of_interest):
                radiances_of_interest[i, j, k1] = radiance[i,j,k2]
                radiances_of_interest_unc[i, j, k1] = radiance_unc[i,j,k2]
            
            if all_data is False:
                ###Take greatest 50% of these radiances, turn rest into zeros, keep x and y locations of everything    
                mean_radiance = np.nanmean(radiances_of_interest[i, j])
                mean_radiance_unc = np.nanmean(radiances_of_interest_unc[i, j])
                for k, rad in enumerate(radiances_of_interest):
                    if radiances_of_interest[i,j,k] < mean_radiance:
                        radiances_of_interest[i,j,k] = 0.0
                        radiances_of_interest_unc[i,j,k] = 0.0
                        
                        
            brightnesses[i,j] = 0.04*radiances_of_interest[i, j, :].sum()  
            brightnesses_unc[i,j] = 0.04*math.sqrt((radiances_of_interest_unc[i, j, :]**2).sum())
    
    return brightnesses, brightnesses_unc
                




            