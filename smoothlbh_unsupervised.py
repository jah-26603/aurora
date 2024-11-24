# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:09:45 2024

@author: JDawg
"""
import functions 
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet
import os
import glob
from tqdm import tqdm
import cv2
from time import time as timer
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import gaussian_filter1d
import scipy.signal
import pandas as pd


def results_loop(time_of_scan, difference_LBHS, border_image, keys, file, latitude, lat_threshold, day, brightnesses_LBHS, sides, graphic_outputs):
        a,b = sides
        difference_LBHS_plot = np.copy(difference_LBHS)
        difference_LBHS_plot[:,int(a)] = np.nan
        difference_LBHS_plot[:,int(b)] = np.nan
        difference_LBHS = difference_LBHS*border_image
        dp = difference_LBHS.astype(np.float32)
        gk1 = functions.gabor_fil(int(keys[file].split('_')[0]))
        filtered_image, kernel = functions.LoG_filter_opencv(dp, sigma_x = .65, sigma_y =.35, size_x = 7, size_y = 5)
        filtered_image = cv2.convertScaleAbs(filtered_image)


        filtered_image = np.abs(cv2.filter2D(filtered_image, -1, gk1).astype(float))
        filtered_image[filtered_image == 0] = np.nan
        filtered_image[filtered_image < np.nanmedian(filtered_image)] = np.nan
        filtered_image[filtered_image < np.nanmedian(filtered_image)] = np.nan
        filtered_image[~np.isnan(filtered_image)] = 1
        

        show_plots = False

        try:
            results = functions.clustering_routine(dp, filtered_image, difference_LBHS, latitude, graphic_outputs, lat_threshold = lat_threshold)
            functions.save_array(difference_LBHS_plot, day, time_of_scan,'difference', graphic_outputs['difference'], show_plots = show_plots)
            functions.save_array(results, day, time_of_scan,'results', graphic_outputs['results'], cmap = 'plasma', show_plots = show_plots)
            functions.save_array(brightnesses_LBHS, day, time_of_scan,'raw_north', graphic_outputs['raw_north'], show_plots = show_plots)

        except ValueError:
            print('no points meet criteria in this scan')
            functions.save_array(difference_LBHS, day, time_of_scan,'difference', graphic_outputs['difference'], show_plots = show_plots)
            functions.save_array(brightnesses_LBHS, day, time_of_scan,'raw_north', graphic_outputs['raw_north'], show_plots = show_plots)
            functions.save_array(np.zeros((53, 92)), day, time_of_scan,'results', graphic_outputs['results'], cmap = 'plasma', show_plots = show_plots)
        

def process_loop(start_file, minus_end_file, north_day, south_day, north_filepath, south_filepath, graphic_outputs):
    file_list = glob.glob(f'{north_filepath}/*.nc')
    south_LBHS = np.zeros((53, 92))
    hemisphere_order = []
    lat_threshold = 50
    dict_list_south_scans = functions.time_window(north_filepath, south_filepath, time_window = (-20, 20))
    keys = list(dict_list_south_scans.keys())
    count = 0
    
    points = np.zeros((len(keys),2))
    image_array = np.zeros((len(keys),52,92))
    difference_LBHS_array = np.zeros_like(image_array)
    border_image_array = np.zeros_like(image_array)
    diff_array = [] * len(keys)
    second_der_array = [] * len(keys)
    opoints = np.zeros((len(keys),2))
    time_of_scan_array = []
    day_array =[]
    brightnesses_LBHS_array = np.zeros_like(image_array)
    file_arr = []
    for file in tqdm(range(start_file, int(len(keys))- minus_end_file)):
        
        # if count%2 == 0:
        #     file = -file
            
        # print(count)
        try:
            ds = nc.Dataset(file_list[file], 'r')
        except OSError:
            print('Error in reading netcdf4 file')
            continue
        if count == 0:
            day = ds.DATE_START[:10]
            print('\n')
            print('Northern Hemisphere Scans from:   ' + day)

        count +=.5

        latitude = ds.variables['GRID_LAT'][:]    #Note: latitude data is flipped!! So data at top is southern most latitude and data at bottom is Northern most latitude
        time = ds.variables['TIME_UTC'][:]
        radiance = ds.variables['RADIANCE'][:]
        wavelength = ds.variables['WAVELENGTH'][:]
        radiance = np.clip(radiance, 0, np.inf)
        sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]




        hemisphere_order, hemisphere, skip_s, skip_n = functions.hemisphere(hemisphere_order, sza, skip_s = True, skip_n = False) #which hemisphere  
        if skip_s == 1 and hemisphere_order[-1] == 1:
            continue
        if skip_n == 1 and hemisphere_order[-1] == 0:
            continue


        filled_indices, one_pixel = functions.filled_indices(wavelength) #acceptable indices for analysis
        date, time_array =  functions.date_and_time(filled_indices, time) #gets date and time
        time_of_scan = time_array[91,81]
        brightnesses_LBHS = functions.get_data_info(radiance, one_pixel, 138, 152, 148, 150, multi_regions= True)        

        
        south_LBHS = functions.get_south_half(dict_list_south_scans[keys[file]], brightnesses_LBHS)
        if hemisphere_order[-1] == 1:
            continue
    
        'Work in progress...'
        #This applies a background mask to get rid of stars so their intensities aren't considered
        background_mask = np.ones_like(latitude)
        background_mask[np.isnan(latitude)] = 0 #removes stars
        brightnesses_LBHS = (brightnesses_LBHS*background_mask)[52:]
        south_LBHS = (south_LBHS*background_mask)[:52]
        try:
            difference_LBHS, plot_diff, diff = functions.absolute_difference(brightnesses_LBHS, south_LBHS)
        except ValueError:
            print("Missing result from", time_of_scan )
            continue
        border_image, lb, rb, dummy_diff, dummy_second_der, dminv, dmaxv = functions.find_edge(difference_LBHS, diff, latitude)
        
        points[file] = [lb, rb]
        opoints[file] = [dminv,dmaxv]
        diff_array.append(dummy_diff)
        second_der_array.append(dummy_second_der)
        time_of_scan_array.append(time_of_scan)
        difference_LBHS_array[file] = difference_LBHS
        day_array.append(day)
        brightnesses_LBHS_array[file] = brightnesses_LBHS
        file_arr.append(file)
    from scipy.signal import medfilt
    
    
    pp = points[~(points == 0).all(axis=1)]
    op = opoints[~(opoints == 0).all(axis=1)]
    difference_LBHS_array = difference_LBHS_array[~(difference_LBHS_array == 0).all(axis=2)]
    difference_LBHS_array = difference_LBHS_array.reshape(difference_LBHS_array.shape[0]//52, 52, 92)
    brightnesses_LBHS_array = brightnesses_LBHS_array[~(brightnesses_LBHS_array == 0).all(axis=2)]
    brightnesses_LBHS_array = brightnesses_LBHS_array.reshape(brightnesses_LBHS_array.shape[0]//52, 52, 92)
    
    plt.figure()
    plt.plot(medfilt(pp[:,1],7), color = 'black', label = 'Smooth Decision Boundaries')
    plt.plot(medfilt(pp[:,0],7), color = 'black')
    plt.plot(pp, color = 'red', label = 'OG Decision Boundaries')
    plt.title('Decision Boundary vs file number (time)')
    plt.ylabel('Column Boundaries for limb exclusion')
    plt.legend()
    plt.show()
    smooth_points = np.zeros_like(pp)
    smooth_points[:,1] = medfilt(pp[:,1],7)
    smooth_points[:,0] = medfilt(pp[:,0],7)
    actual_points = np.copy(pp)
    c, r = np.where(np.abs(smooth_points- pp) > 3)
    actual_points[c,r] = smooth_points[c,r]
        
    for i, file in tqdm(enumerate(file_arr)):
        dummy = np.copy(latitude)
        dummy[np.isnan(dummy)] = 0
        dummy[dummy != 0 ] = 1
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])

        neighbor_check = cv2.filter2D(dummy, -1, kernel)
        border_image = np.where((neighbor_check < 8) & (dummy == 1), 1, 0)       #perhaps change to 7?
        border_image = border_image[52:]
        border_image = np.abs(border_image - 1)
        border_image[:,int(pp[i,0]) :int(pp[i,1])] = 1
        results_loop(time_of_scan_array[i], difference_LBHS_array[i], border_image, 
                     keys, file_arr[i], latitude, lat_threshold, day_array[i],
                     brightnesses_LBHS_array[i], actual_points[i], graphic_outputs)

    # for i in range(image_array.shape[0]):
    #     fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # Adjust `figsize` as needed
    
    #     # Display the image on the first subplot
    #     axes[0].imshow(image_array[i], aspect='auto')
    #     axes[0].axvline(x=pp[i, 0], color='white')
    #     axes[0].axvline(x=pp[i, 1], color='white')
    #     axes[0].axis('off')  
    
    #     # Plot the data on the second subplot
    #     axes[1].plot(second_der_array[i] + 2, label='Second Derivative')
    #     axes[1].plot(diff_array[i], label='Difference Array')
    #     axes[1].axvline(x=op[i, 0], color='red')
    #     axes[1].axvline(x=op[i, 1], color='red')
    #     axes[1].legend()
    #     axes[1].set_xlim([15, 124-16])  
    #     # axes[1].axis('off')
              
    #     # Adjust layout and show the figure
    #     plt.tight_layout()
    #     plt.show()
    
if __name__ == "__main__":
    north_day =  "C:\\Users\\JDawg\\Desktop\\Aurora_Dates\\2020\\295"
    south_day = "C:\\Users\\JDawg\\Desktop\\Aurora_Dates\\2021\\112"
    
    north_filepath = f"{north_day}\\data"
    south_filepath = f"{south_day}\\data"
    
    graphic_outputs = {
        'raw_north': f"{north_day}\\graphics\\raw_north",
        'difference': f"{north_day}\\graphics\\difference",
        'results': f"{north_day}\\graphics\\results"
    }

    process_loop(north_day, south_day, north_filepath, south_filepath, graphic_outputs)