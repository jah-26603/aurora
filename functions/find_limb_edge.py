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
def find_edge(difference_LBHS, diff, latitude):
    
    
    
        dummy = np.copy(latitude)
        dummy[np.isnan(dummy)] = 0
        dummy[dummy != 0 ] = 1
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        
        neighbor_check = cv2.filter2D(dummy, -1, kernel)
        pp = np.copy(latitude)
        pp[np.isnan(pp)] = 0
        pp[pp!= 0 ] = 1
        nb = cv2.filter2D(pp, -1, kernel)
        bi = np.where((nb < 7) & (pp == 1), 1, np.nan)      
        bi = bi[52:]

        sig = 5
        second_der = gaussian_laplace(diff, sigma=sig)/ np.max(np.abs(gaussian_laplace(diff, sigma=sig)))
        zrs = np.where(np.diff(np.sign(second_der)) != 0)[0]

        min_v = 30
        max_v = 95

        peaks = scipy.signal.find_peaks(second_der, prominence = 0.1, distance = 5 )
        o_peaks = scipy.signal.find_peaks(-second_der, prominence = 0.1, distance = 5 )[0]
        left_peak = peaks[0][np.min(np.where(peaks[0] > min_v))] 
        right_peak = peaks[0][np.max(np.where(peaks[0] < max_v))]
        o_peaks = o_peaks[np.logical_and(o_peaks > left_peak, o_peaks < right_peak)]

        valid_range = zrs[np.where((zrs > left_peak) & (zrs < right_peak))]
        col, row = (np.where(~np.isnan((difference_LBHS * bi).T)))

        grad_array = np.diff(second_der)
        try:
            min_v = np.min(valid_range[np.where(grad_array[valid_range] <0)]) 
        except ValueError or UnboundLocalError:
            breakpoint()
        
        max_v = np.max(valid_range[np.where(grad_array[valid_range] >0)]) + 1
        
        if min_v > np.min(o_peaks):
            min_v = left_peak + int((np.min(o_peaks) - left_peak) /2) 
        if max_v < np.max(o_peaks):
            max_v = right_peak + int((np.max(o_peaks) - right_peak) /2) 
            
        plt.figure()
        plt.plot((diff/np.max(diff)))
        plt.axvline(x = min_v, color = 'red', label = 'axvline - full height')
        plt.axvline(x = max_v, color = 'red', label = 'axvline - full height')
        plt.plot(second_der)
        plt.plot(diff/np.max(diff) * -second_der/ np.max(-second_der) + 2 )
        plt.show()
        
        border_image = np.where((neighbor_check < 8) & (dummy == 1), 1, 0)      
        border_image = border_image[52:]
        border_image = np.abs(border_image - 1)
        border_image[:, col[min_v] :col[max_v]] = 1

        return border_image, col[min_v], col[max_v], diff/np.max(diff),  diff/np.max(diff) * -second_der/ np.max(-second_der), min_v, max_v