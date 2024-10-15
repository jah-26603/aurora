# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:04:18 2024

@author: dogbl
"""
import functions 
import netCDF4 as nc
import numpy as np


# need the file list within a given time window
# need to do some operation on the arrays in south_scans

def get_south_half(file_list, north, time_window = 30):
    hemisphere_order = []
    south_scans = []
    
    for file in file_list:
        ds = nc.Dataset(file, 'r')
        radiance = ds.variables['RADIANCE'][:]
        sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
        wavelength = ds.variables['WAVELENGTH'][:]
    

        radiance = np.clip(radiance, 0, np.inf)
        hemisphere_order, hemisphere, skip_s, skip_n = functions.hemisphere(hemisphere_order, sza, skip_s = False, skip_n = True, print_b = False) #which hemisphere
        
        if skip_s == 1 and hemisphere_order[-1] == 1:
            continue
        if skip_n == 1 and hemisphere_order[-1] == 0:
            continue
        
    
        filled_indices, one_pixel = functions.filled_indices(wavelength) #acceptable indices for analysis
        brightnesses_LBHS = functions.get_data_info(radiance, one_pixel, 138, 152, 148, 150, multi_regions= True)    
        south_scans.append(brightnesses_LBHS)
    

    resultant_south_scan = np.nanmean(np.array(south_scans), axis = 0)
    
    return resultant_south_scan
    
    
    
    