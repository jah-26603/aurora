# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:09:45 2024

@author: JDawg
"""
import functions 
import netCDF4 as nc
import numpy as np
import cartopy
import torch
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import cm
import pylab as pl
import statistics
import random
from tabulate import tabulate
import cartopy
from cartopy import config
import cartopy.crs as ccrs
import matplotlib.pylab as plb
from matplotlib import colorbar, colors
from PIL import Image
import glob
from datetime import datetime
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans


file_path = r".\Aurora_Dates\03_21_2021"
# functions.read_tar(file_path)
file_list = glob.glob(f'{file_path}/*.nc')

dims_ang = (91, 91) 
dims_fl = (104,92)


all_O, n_all_O, all_LBH, n_all_LBH, access, access_n = [np.zeros((len(file_list), *dims_ang)) for _ in range(6)]
all_Ob, all_brightnesses, all_sza, all_emission_angle, all_longitude, all_latitude =[np.empty((len(file_list), *dims_fl)) for _ in range(6)]
all_time = []
hemisphere_order = []
n_indices_LBH  = [[[[] for _ in range(91)] for _ in range(91)] for _ in range(len(file_list))]
lat_vals = [[[] for _ in range(91)] for _ in range(91)]
long_vals = [[[] for _ in range(91)] for _ in range(91)]
def fittin(y, a,b,c ,d):
    return (a*y+b*y**.5+ c +d*y**.75)



for file in range(len(file_list)):
# for file in range(1,3):
    ds = nc.Dataset(file_list[file], 'r')
    
    #store data in arrays
    longitude = ds.variables['GRID_LON'][:]
    latitude = ds.variables['GRID_LAT'][:]    #Note: latitude data is flipped!! So data at top is southern most latitude and data at bottom is Northern most latitude
    time = ds.variables['TIME_UTC'][:]
    radiance = ds.variables['RADIANCE'][:]
    radiance_unc = ds.variables['RADIANCE_RANDOM_UNC'][:]  #are these the uncertainties?
    sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
    wavelength = ds.variables['WAVELENGTH'][:]
    emission_angle = ds.variables['EMISSION_ANGLE'][:]
    
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
    
    
    functions.plot_on_globe(latitude, longitude, brightnesses_LBHS, date, time_array, filled_indices, 'LBHS', hemisphere_order, skip_south_plot = True, vmax = 1000)
    brightnesses_LBHS = np.nan_to_num(brightnesses_LBHS)
    

        # Range of cluster numbers to try
    k_range = range(1, 12)  # For example, testing k from 1 to 10
    sse = []
    
    # Calculate SSE for each number of clusters
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(brightnesses_LBHS)
        sse.append(kmeans.inertia_)
    
    # Plot the elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.show()
    # Step 1: Reshape the data to (num_pixels, 1)
    original_shape = brightnesses_LBHS.shape
    data = brightnesses_LBHS.reshape(-1, 1)

    # Step 2: Apply KMeans clustering
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_

    # Step 3: Reshape the clustered data back to the original image shape
    segmented_image = labels.reshape(original_shape)

    # # Step 4: Plot the segmented image
    # plt.figure(figsize=(10, 5))
    # plt.imshow(segmented_image, cmap='viridis', interpolation='none')
    # plt.colorbar()
    # plt.title('Segmented Image with 3 Regions')
    # plt.show()
    
    
    functions.plot_on_globe(latitude, longitude, segmented_image, date, time_array, filled_indices,  'LBHS', hemisphere_order, skip_south_plot = True, vmax = n_clusters)





     

