# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:57:40 2024

@author: JDawg
"""
import numpy as np

def filled_indices(wavelength):
    #Find i,j indices of wavelengths that are filled with data
    filled_indices = []
    for i in range(wavelength.shape[0]):
        for j in range(wavelength.shape[1]):
            if np.isfinite(wavelength[i][j]).any():
                filled_indices.append((i, j))
    filled_indices = np.array(filled_indices)
    one_pixel = np.array(wavelength[filled_indices[0][0],filled_indices[0][1],:]) #array of 800 wavelength values for one pixel
    
    
    return filled_indices, one_pixel