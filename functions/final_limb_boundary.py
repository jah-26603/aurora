# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:45:20 2025

@author: JDawg
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import scipy
import scipy.signal
import cv2
from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt

def find_boundary(zrs, peaks, left_region, right_region):
    def get_max_boundary(region):
        zz = np.intersect1d(zrs, region)
        return np.max(zz) if zz.size else np.max(peaks)

    lb = get_max_boundary(left_region)
    rb = get_max_boundary(right_region)

    return lb, rb

def final_limb_boundary(species_datastructure, estimated_limb_boundary, latitude):
    

    search_radius = 7
    limb_boundary = {}
    for specie in list(species_datastructure.keys()):
        limb_boundary[specie] = {'points': []}
        for i in range(len(estimated_limb_boundary)):
            
            dummy = np.where(np.isnan(latitude), 0, 1)
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]])
            
            nb = cv2.filter2D(dummy.astype(np.float32), -1, kernel)
            bi = np.where((nb < 7) & (dummy == 1), 1, np.nan)[52:]
            col, row = np.where(~np.isnan((bi).T))
            neighbor_check = cv2.filter2D(dummy.astype(np.float32), -1, kernel)
            
            
            lb,rb = estimated_limb_boundary[i] #initial guesses of limb boundaries
            bi[:,int(lb)] +=1
            bi[:,int(rb)] +=1
            k1 = ((bi).T).flatten()
            k2 = k1[~np.isnan(k1)]
            boundaries = np.where(k2 == 2)[0]
            if len(boundaries) > 2:
                breakpoint()


            dummy = species_datastructure[specie]['diff']
            diff = dummy[i]
            second_der = species_datastructure[specie]['second_der'][i]
            
            zrs = np.where(np.diff(np.sign(second_der)) != 0)[0]
            peaks = scipy.signal.find_peaks(second_der, prominence = 0.1, distance = 5)[0]
            
            left_region = np.arange(boundaries[0] - search_radius, boundaries[0] + search_radius + 1)
            right_region = np.arange(boundaries[1] - search_radius, boundaries[1] + search_radius + 1)
            

            zz = np.intersect1d(zrs, left_region)
            pp = np.intersect1d(peaks, left_region)
            
            if len(zz) != 0:
                lb = np.min(zz)
            elif len(pp) != 0 :
                lb = np.min(pp)

            zz = np.intersect1d(zrs, right_region)
            pp = np.intersect1d(peaks, right_region)
            
            if len(zz) != 0:
                rb = np.max(zz)
            elif len(pp) != 0:
                rb = np.max(pp)

            bb = [col[int(lb)],col[int(rb)]]
            limb_boundary[specie]['points'].append(bb)
            plt.figure()
            plt.plot(diff)
            plt.plot(second_der)
            plt.axvline(lb)
            plt.axvline(rb)
            plt.show()
        limb_boundary[specie]['points'] = np.array(limb_boundary[specie]['points'])

    return limb_boundary