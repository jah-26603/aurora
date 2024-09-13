# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:09:45 2024

@author: JDawg
"""
import functions 
import netCDF4 as nc
import numpy as np
import torch
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
from tqdm import tqdm

file_path = r".\Aurora_Dates\03_19_2020_wk"
# functions.read_tar(file_path)
file_list = glob.glob(f'{file_path}/*.nc')
south_LBHS = np.zeros((53, 92))
south_O = np.zeros((53, 92))
hemisphere_order = []
lat_threshold = 50

for file in tqdm(range(len(file_list))):
    
    
    ds = nc.Dataset(file_list[file], 'r')
    #store data in arrays
    longitude = ds.variables['GRID_LON'][:]
    latitude = ds.variables['GRID_LAT'][:]    #Note: latitude data is flipped!! So data at top is southern most latitude and data at bottom is Northern most latitude
    time = ds.variables['TIME_UTC'][:]
    radiance = ds.variables['RADIANCE'][:]
    radiance_unc = ds.variables['RADIANCE_RANDOM_UNC'][:]  #are these the uncertainties?
    sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
    wavelength = ds.variables['WAVELENGTH'][:]

    
    radiance = np.clip(radiance, 0, np.inf)
    hemisphere_order, hemisphere, skip_s, skip_n = functions.hemisphere(hemisphere_order, sza, skip_s = False, skip_n = False) #which hemisphere
    
    if skip_s == 1 and hemisphere_order[-1] == 1:
        continue
    if skip_n == 1 and hemisphere_order[-1] == 0:
        continue
    
    
    
    filled_indices, one_pixel = functions.filled_indices(wavelength) #acceptable indices for analysis
    date, time_array =  functions.date_and_time(filled_indices, time) #gets date and time
    
    brightnesses_LBHS, brightnesses_LBHS_unc = functions.get_data_info(radiance, radiance_unc, one_pixel, 138, 152, 148, 150, multi_regions= True)    
    brightnesses_O, brightnesses_O_unc = functions.get_data_info(radiance, radiance_unc, one_pixel, 135, 136, np.nan, np.nan)
    
    #Just trying to plot without nan data, figure it out later....
    if hemisphere_order[-1] == 1:
        south_LBHS = brightnesses_LBHS[:53]
        south_O = brightnesses_O[:53]
        continue



    difference_LBHS = functions.absolute_difference(brightnesses_LBHS[51:], south_LBHS)
    difference_O = functions.absolute_difference(brightnesses_O[51:], south_O)
    segmented_image_LBHS, n_clusters, kmeans_LBHS = functions.segment_image(difference_LBHS)
    segmented_image_O, _, kmeans_O = functions.segment_image(difference_O)
    
    
    reshaped_labels = np.reshape(np.array(kmeans_LBHS.labels_), difference_O.shape)
    masks = [reshaped_labels == group for group in range(n_clusters)]
    combined_mask = np.zeros_like(masks[0], dtype=bool)  # Start with all False
    threshold = np.array([arr * latitude[51:] for arr in masks])
    threshold[threshold==0] = np.nan
    
    for i in range(len(threshold)):
        if np.nanmedian(threshold[i]) > lat_threshold:
            combined_mask |= masks[i]  # Logical OR to combine
            
    segmented_image_LBHS = combined_mask * segmented_image_LBHS
    images = [brightnesses_LBHS[51:], difference_LBHS, segmented_image_LBHS, segmented_image_O]
    labels = ['LBHS Raw (R)', 'LBHS difference (R)', 'Segmented LBHS', 'Segmented O']
    levels = [1000, 1000, n_clusters, n_clusters]
    functions.plot_on_globe(latitude[51:], longitude[51:], images, date, time_array, filled_indices, labels, hemisphere_order, skip_south_plot = True, vmax = levels)





