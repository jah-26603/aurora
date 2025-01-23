# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:35:37 2024

@author: JDawg
"""
import numpy as np
import functions
import cv2
import matplotlib.pyplot as plt

def results_loop(time_of_scan, difference, border_image, keys, file, latitude, 
                 lat_threshold, day, brightnesses, sides, graphic_outputs, species):
        a,b = sides
        difference_plot = np.copy(difference)
        
        difference_plot[:,int(a)] = np.nan
        difference_plot[:,int(b)] = np.nan
        # difference = difference*border_image
        difference = difference #comment out for limb detection
        
        dp = difference.astype(np.float32)
        gk1 = functions.gabor_fil(int(keys[file].split('_')[0]))
        filtered_image, kernel = functions.LoG_filter_opencv(dp, sigma_x = .65, sigma_y =.35, size_x = 7, size_y = 5)
        filtered_image = cv2.convertScaleAbs(filtered_image)


        filtered_image = np.abs(cv2.filter2D(filtered_image, -1, gk1).astype(float))
        filtered_image[filtered_image == 0] = np.nan
        filtered_image[filtered_image < np.nanmedian(filtered_image)] = np.nan
        filtered_image[filtered_image < np.nanmedian(filtered_image)] = np.nan
        filtered_image[~np.isnan(filtered_image)] = 1
        

        show_plots = True
        functions.save_array(difference_plot, day, time_of_scan,'difference', graphic_outputs['difference'], species, show_plots = show_plots)
        functions.save_array(brightnesses, day, time_of_scan,'raw_north', graphic_outputs['raw_north'], species, show_plots = show_plots)

        try:
            results = functions.clustering_routine(dp, filtered_image, difference, latitude, graphic_outputs, lat_threshold = lat_threshold)
            
            results *= border_image
            results[:,int(a)] = np.nan
            results[:,int(b)] = np.nan
            functions.save_array(results, day, time_of_scan,'results', graphic_outputs['results'], species, cmap = 'plasma', show_plots = show_plots)
        except ValueError:
            print('no points meet criteria in this scan')
            functions.save_array(np.zeros((53, 92)), day, time_of_scan,'results', graphic_outputs['results'], cmap = 'plasma', show_plots = show_plots)
        
