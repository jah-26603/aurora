# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:11:10 2024

@author: dogbl
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
import pandas as pd
import scipy.signal
from scipy.signal import medfilt
from sklearn.decomposition import FastICA
def absolute_difference(north, south, latitude):
    south = np.fliplr(np.rot90(south, k = 2))

    a = 0
    b = 20
    c = 8
    d = 73
    n_f = np.nanmedian(north[a:b,c:d])
    s_f = np.nanmedian(south[a:b,c:d])

    if s_f == 0:
        s_f = n_f
    difference = north- (n_f/s_f)*south
    difference = np.clip(difference, 0, np.inf)
    
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    lat = latitude[52:]
    lat[np.isnan(lat)] = 0
    lat[lat != 0 ]= 1
    nb = cv2.filter2D(lat, -1, kernel)
    bi = np.where((nb <= 6) & (lat == 1), 1, np.nan)      




    # north[north == 0] = np.nan
    # north = north + np.roll(north, shift=(1, 0), axis=(0, 1))
    # north[np.isnan(north)] = 0 
    
    # south[south == 0] = np.nan
    # south = south + np.roll(south, shift=(1, 0), axis=(0, 1))
    # south[np.isnan(south)] = 0

    #Need to refactor this segment at some point

    kk1 = ((north*bi).T).flatten()
    kk2 = kk1[~np.isnan(kk1)]
    ss1 = (south*bi).T.flatten()
    ss2 = np.copy(ss1)[~np.isnan(ss1)]        
    ss3 = gaussian_filter1d(ss2, sigma = 7)

    #scaling for du?
    kk3 = kk2/np.max(np.concatenate((kk2[:50], kk2[100:])))
    du = (kk3 / np.max(kk3))
    du1 = np.concatenate(((kk3 / np.max(kk3))[:50], (kk3 / np.max(kk3))[100:]))
    su =  ss3 / np.max(ss3)
    su1 = np.concatenate((su[:50], su[100:]))
    suu = np.clip(su - np.mean(su1) + 1*np.mean(du1), 0 ,1) +.01
    diff = np.clip((du/np.max(du)  - suu/np.max(suu)), 0,1)
    diff/= np.max(diff)
    
    
    from scipy.signal import correlate



    sig = 5
    peaks = scipy.signal.find_peaks(medfilt(du, sig),  distance = 3 )    
    # plt.figure(figsize=(10, 5))
    # plt.plot(du, label="Noisy Line", color="red")
    # plt.plot(medfilt(du, kernel_size = sig))
    # plt.plot(suu, label="Original Smooth Line", color="blue")
    # plt.scatter(peaks[0], du[peaks[0]], color = 'red')
    # plt.legend()
    # plt.show()
    

    
    du = medfilt(du, kernel_size = sig)
    result = np.clip(du - suu, 0, 1 )/ np.max(np.clip(du - suu, 0, 1 ))
    

    # # Stack the data into a matrix (rows are samples, columns are signals)
    # data = np.vstack([du, su]).T
    # dataa = np.vstack([du, suu]).T
    # ica = FastICA(n_components=2, random_state=42)
    # # plt.figure()
    # # plt.plot(ica.fit_transform(data))
    # # plt.show()
    
    # # plt.figure()
    # # plt.plot(ica.fit_transform(dataa))
    # # plt.show()
    
    return difference, result

