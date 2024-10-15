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
from time import time as timer



def process_loop(north_day, south_day, north_filepath, south_filepath, graphic_outputs):
    file_list = glob.glob(f'{north_filepath}/*.nc')
    south_LBHS = np.zeros((53, 92))
    hemisphere_order = []
    lat_threshold = 50
    dict_list_south_scans = functions.time_window(north_filepath, south_filepath, time_window = (-20, 40))
    keys = list(dict_list_south_scans.keys())
    count = 0
    
    
    for file in tqdm(range(50, len(keys) - 0)):
        try:
            ds = nc.Dataset(file_list[file], 'r')
        except OSError:
            print('Error in reading netcdf4 file')
            continue
        if count == 0:
            day = ds.DATE_START[:10]
            print('\n')
            print('Northern Hemisphere Scans from:   ' + day)
            count +=1
        
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
        

        #This applies a background mask to get rid of stars so their intensities aren't considered
        background_mask = np.ones_like(latitude)
        background_mask[np.isnan(latitude)] = 0 #removes stars
        brightnesses_LBHS = (brightnesses_LBHS*background_mask)[52:]
        south_LBHS = (south_LBHS*background_mask)[:52]
        difference_LBHS = functions.absolute_difference(brightnesses_LBHS, south_LBHS)
        
        if int(time_of_scan[:2]) <= 13:
            a = 24
            b = 57
        elif int(time_of_scan[:2]) < 17:
            a = 32
            b = 59
        else:
            a = 31
            b = 61
        

        dummy = np.copy(latitude)
        dummy[np.isnan(dummy)] = 0
        dummy[dummy != 0 ] = 1
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        
        neighbor_check = cv2.filter2D(dummy, -1, kernel)
        border_image = np.where((neighbor_check < 8) & (dummy == 1), 1, 0)      
        border_image = border_image[52:]
        border_image = np.abs(border_image - 1)
        border_image[:, a:b] = 1
        breakpoint()
        difference_LBHS = difference_LBHS*border_image

        
        dp = difference_LBHS.astype(np.float32)
        gk1 = functions.gabor_fil(int(keys[file].split('_')[0]))
        # gk1 = functions.gabor_fil(int(keys[file].split('_')[0]), theta = 45, storm = True)
        filtered_image, kernel = functions.LoG_filter_opencv(dp, sigma_x = .65, sigma_y =.35, size_x = 7, size_y = 5)
        filtered_image = cv2.convertScaleAbs(filtered_image)
        plt.imshow(filtered_image)
        filtered_image = np.abs(cv2.filter2D(filtered_image, -1, gk1).astype(float))
        filtered_image[filtered_image == 0] = np.nan
        filtered_image[filtered_image < np.nanmedian(filtered_image)] = np.nan
        filtered_image[filtered_image < np.nanmedian(filtered_image)] = np.nan
        filtered_image[~np.isnan(filtered_image)] = 1
        

        show_plots = True    
        try:
            results = functions.clustering_routine(dp, filtered_image, difference_LBHS, latitude, graphic_outputs, lat_threshold = lat_threshold)
            functions.save_array(difference_LBHS, day, time_of_scan,'difference', graphic_outputs['difference'], show_plots = show_plots)
            functions.save_array(results, day, time_of_scan,'results', graphic_outputs['results'], cmap = 'plasma', show_plots = show_plots)
            # functions.save_array(np.fliplr(np.rot90(south_LBHS, k = 2)), day, time_of_scan,'south', graphic_outputs['raw_north'], show_plots = show_plots)
            functions.save_array(brightnesses_LBHS, day, time_of_scan,'raw_north', graphic_outputs['raw_north'], show_plots = show_plots)

        except ValueError:
            print('no points meet criteria in this scan')
            functions.save_array(difference_LBHS, day, time_of_scan,'difference', graphic_outputs['difference'], show_plots = show_plots)
            functions.save_array(brightnesses_LBHS, day, time_of_scan,'raw_north', graphic_outputs['raw_north'], show_plots = show_plots)
            functions.save_array(np.zeros((53, 92)), day, time_of_scan,'results', graphic_outputs['results'], cmap = 'plasma', show_plots = show_plots)
        

    functions.create_gif(output_loc = os.path.dirname(graphic_outputs['raw_north']), **graphic_outputs )



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