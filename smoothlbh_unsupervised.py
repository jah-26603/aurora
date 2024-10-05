# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:09:45 2024

@author: JDawg
"""
import functions 
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import cv2
from sklearn.cluster import DBSCAN

filepath = r"C:\Users\JDawg\Desktop\Aurora_Dates\2020\326"
antipode_filepath = r"C:\Users\JDawg\Desktop\Aurora_Dates\2020\142"


file_list = glob.glob(f'{filepath}/*.nc')
south_LBHS = np.zeros((53, 92))
south_O = np.zeros((53, 92))
hemisphere_order = []
lat_threshold = 50

# d_out = r'C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\outputs\difference'
# r_out = r'C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\outputs\result'


dict_list_south_scans = functions.time_window(filepath, antipode_filepath, time_window = (-20, 20))
keys = list(dict_list_south_scans.keys())
count = 0
for file in tqdm(range(len(keys))):
    ds = nc.Dataset(file_list[file], 'r')

    if count == 0:
        print('\n')
        print('Northern Hemisphere Scans from:   ' + ds.DATE_START[:10])
    count +=1
    
    #store data in arrays
    longitude = ds.variables['GRID_LON'][:]
    latitude = ds.variables['GRID_LAT'][:]    #Note: latitude data is flipped!! So data at top is southern most latitude and data at bottom is Northern most latitude
    time = ds.variables['TIME_UTC'][:]
    radiance = ds.variables['RADIANCE'][:]
    radiance_unc = ds.variables['RADIANCE_RANDOM_UNC'][:]  #are these the uncertainties?
    sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
    wavelength = ds.variables['WAVELENGTH'][:]

    radiance = np.clip(radiance, 0, np.inf)
    hemisphere_order, hemisphere, skip_s, skip_n = functions.hemisphere(hemisphere_order, sza, skip_s = True, skip_n = False) #which hemisphere
    
    if skip_s == 1 and hemisphere_order[-1] == 1:
        continue
    if skip_n == 1 and hemisphere_order[-1] == 0:
        continue
    
    
    
    filled_indices, one_pixel = functions.filled_indices(wavelength) #acceptable indices for analysis
    date, time_array =  functions.date_and_time(filled_indices, time) #gets date and time
    
    brightnesses_LBHS, brightnesses_LBHS_unc = functions.get_data_info(radiance, radiance_unc, one_pixel, 138, 152, 148, 150, multi_regions= True)        
    south_LBHS = functions.get_south_half(dict_list_south_scans[keys[file]], brightnesses_LBHS)
    if hemisphere_order[-1] == 1:
        continue
    
    #This applies a background mask to get rid of stars so their intensities aren't considered
    background_mask = np.ones_like(latitude)
    background_mask[np.isnan(latitude)] = 0 #removes stars
    brightnesses_LBHS = (brightnesses_LBHS*background_mask)[52:]
    south_LBHS = (south_LBHS*background_mask)[:52]
    difference_LBHS = functions.absolute_difference(brightnesses_LBHS, south_LBHS)


        
        
    plt.figure()
    plt.imshow(difference_LBHS)
    plt.title(f'{time_array[filled_indices[0,0], filled_indices[0,1]]}')
    plt.show()
    
    
    dp = difference_LBHS.astype(np.float32)
    
    
    # Parameters for Gabor filter
    ksize = 7  # Size of the Gabor kernel
    sigma = 20  # Standard deviation of the Gaussian function
    theta = np.deg2rad(97)  # Orientation of the Gabor kernel (135 degrees for southwest)
    lambd = 5  # Wavelength of the sinusoidal factor
    gamma = 0.05  # Aspect ratio of the Gaussian function
    psi = 0  # Phase offset

    # Create Gabor kernel
    gk1 = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)

    image = dp  # Replace 'path_to_your_image.png' with your image path
    filtered_image, kernel = functions.LoG_filter_opencv(image,sigma_x = .9, sigma_y =.9, size_x = 7, size_y = 5)
    filtered_image = cv2.convertScaleAbs(filtered_image)
    filtered_image = cv2.filter2D(filtered_image, -1, gk1).astype(float)
    filtered_image[filtered_image == 0] = np.nan
    filtered_image[~np.isnan(filtered_image)] = 1


    qq = dp* filtered_image
    qq[qq< np.nanmedian(qq)] = np.nan
    qq[np.isnan(qq)] = 0 
    qq[qq!=0] = 1

    
    
    # #CLUSTERING
    i, j = np.indices(qq.shape)
    vector = np.column_stack((i.ravel(), j.ravel(), qq.ravel()))
    vector= vector[qq.ravel() != 0]
    X = vector[:, :2]  # Use only x and y (i, j) coordinates for spatial clustering

    # Apply DBSCAN
    dbscan = DBSCAN(eps=4, min_samples=6)
    labels = dbscan.fit_predict(X)
    aurora_list = []
    latitude = latitude[52:]
    # First loop: Filter clusters based on latitude condition and store in aurora_list
    for i in range(int(max(labels) + 1)):
        class_vector = X[labels == i]
        row_indices = class_vector[:, 0].astype(int)  # Assuming rows are in the first column
        col_indices = class_vector[:, 1].astype(int)  # Assuming cols are in the second column
    
        # Check if the cluster's median latitude exceeds the threshold
        if np.nanmedian(latitude[row_indices, col_indices]) > lat_threshold:
            aurora_list.append(class_vector)
    
    # Second loop: Plot each cluster's points on a copy of the qq array
    for idx, vec in enumerate(aurora_list):
        row_indices = vec[:, 0].astype(int)  # Assuming rows are in the first column
        col_indices = vec[:, 1].astype(int)  # Assuming cols are in the second column
    
        # Create a copy of qq to visualize each cluster
        dummy = np.zeros_like(qq)
        
        # Mark the points from this cluster in the 'dummy' array
        dummy[row_indices, col_indices] = 1  # Use a distinct value to mark these points
    
        # Plot the result for this cluster
        plt.figure()
        plt.imshow(dummy * difference_LBHS, cmap='plasma')  # Adjust cmap as needed
        plt.title(f'Cluster {idx + 1}')
        plt.show()
    
    
    # d_dummy = np.copy(dp).astype(np.uint8)
    # r_dummy = np.copy(dummy*difference_LBHS).astype(np.uint8)
    
    # from PIL import Image
    # dim = Image.fromarray(d_dummy)
    # rim = Image.fromarray(r_dummy)
    
    # dim.save(rf"{d_out}\{file}.png")
    # rim.save(rf"{r_out}\{file}.png")
    