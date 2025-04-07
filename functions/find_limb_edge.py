# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:57:26 2024

@author: JDawg
"""
import numpy as np
import scipy
import scipy.signal
import cv2
from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt
def find_edge(difference, diff, latitude, sza):
    
        sza[sza<104] = np.nan

        # Identify boundary pixels with fewer than 7 neighbors
        dummy = np.where(np.isnan(latitude), 0, 1)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        
        nb = cv2.filter2D(dummy.astype(np.float32), -1, kernel)
        bi = np.where((nb < 7) & (dummy == 1), 1, np.nan)[52:]
        neighbor_check = cv2.filter2D(dummy.astype(np.float32), -1, kernel)
        
        # Compute second derivative and detect zero crossings across limb
        sig = 5
        second_der = gaussian_laplace(diff, sigma=sig)
        second_der /= np.max(np.abs(second_der))
        zrs = np.where(np.diff(np.sign(second_der)) != 0)[0]
        col, row = np.where(~np.isnan((difference * bi).T))
        min_v = 27
        max_v = 95

        peaks = scipy.signal.find_peaks(second_der, prominence = 0.08, distance = 5 )
        o_peaks = scipy.signal.find_peaks(-second_der, prominence = 0.08, distance = 5 )[0]
        peaks = np.asarray(peaks[0])
        try:
            left_peak = peaks[np.min(np.where(peaks > min_v))] 
            right_peak = peaks[np.max(np.where(peaks < max_v))]
        except ValueError:
            left_peak = np.copy(min_v)
            right_peak = np.copy(max_v)
        o_peaks = o_peaks[np.logical_and(o_peaks > left_peak, o_peaks < right_peak)]

        valid_range = zrs[np.where((zrs > left_peak) & (zrs < right_peak))]

        grad_array = np.diff(second_der)
        try:
            min_v = np.min(valid_range[np.where(grad_array[valid_range] <0)]) 

        except ValueError or UnboundLocalError:
            plt.figure()
            plt.plot(second_der)
            plt.scatter(zrs, second_der[zrs])
            plt.show()
            min_v = np.min(zrs[zrs>30])
        try:
            max_v = np.max(valid_range[np.where(grad_array[valid_range] >0)]) + 1
        except ValueError or UnboundLocalError:
            max_v = np.max(zrs[zrs<85])

        
        border_image = np.where((neighbor_check < 8) & (dummy == 1), 1, 0)      
        border_image = border_image[52:]
        border_image = np.abs(border_image - 1)
            
        border_image[:, col[min_v] :col[max_v]] = 1
        
        # plt.figure()
        # plt.plot(diff)
        # plt.axvline(min_v)
        # plt.axvline(max_v)
        # plt.show()
        # # Compute the absolute difference between consecutive values
        # diffs = np.abs(np.diff(diff))

        # # Define a threshold for "drastic change" (adjust as needed)
        # threshold = 0.01  # This can be tuned

        # # Identify indices where changes are small
        # stable_regions = diffs < threshold

        # # Find continuous segments
        # from itertools import groupby
        # from operator import itemgetter

        # stable_indices = np.where(stable_regions)[0]  # Get indices where diff is small
        # groups = []
        # for k, g in groupby(enumerate(stable_indices), lambda i_x: i_x[0] - i_x[1]):
        #     group = list(map(itemgetter(1), g))
        #     if len(group) > 15:  # Minimum length constraint to avoid noise
        #         groups.append(group)

        # # Print the stable regions
        # for g in groups:
        #     print(f"Stable region: indices {g[0]} to {g[-1]}, values {diff[g[0]:g[-1]+1]}")

        # plt.figure()
        # plt.plot(diff)
        # for g in groups:
        #     plt.axvline(g[0], ymin = 0, ymax = 1, color = 'red')
        #     plt.axvline(g[-1], ymin = 0, ymax = 1, color = 'red')
        # plt.show()
        
        return border_image, col[min_v], col[max_v], diff/np.max(diff), second_der, min_v, max_v