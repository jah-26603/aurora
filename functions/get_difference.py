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
def absolute_difference(north, south):
    south = np.fliplr(np.rot90(south, k = 2))

    a = 0
    b = 20
    c = 8
    d = 73
    n_f = np.nanmedian(north[a:b,c:d])
    s_f = np.nanmedian(south[a:b,c:d])

    difference = north- (n_f/s_f)*south

    difference = np.clip(difference, 0, np.inf)
    plot_diff = np.copy(difference)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    pp = np.copy(north)
    pp[np.isnan(pp)] = 0
    pp[pp!= 0 ] = 1
    nb = cv2.filter2D(pp, -1, kernel)
    bi = np.where((nb <= 6) & (pp == 1), 1, np.nan)      

    

    ll = np.copy(north)
    # ll[ll.mask] = 0



    kk1 = ((ll*bi).T).flatten()
    # kk1 = ((ll*bi)[np.where(~np.isnan((difference * bi)))]).T.flatten()
    kk2 = kk1[~np.isnan(kk1)]
    ss1 = (south*bi).T.flatten()
    ss2 = np.copy(ss1)[~np.isnan(ss1)]        
    ss3 = gaussian_filter1d(ss2, sigma = 7)
    # if len(kk2)> 179:
    # breakpoint()

    #scaling for du?
    kk3 = kk2/np.max(np.concatenate((kk2[:50], kk2[100:])))
    du = (kk3 / np.max(kk3))
    du1 = np.concatenate(((kk3 / np.max(kk3))[:50], (kk3 / np.max(kk3))[100:]))
    su =  ss3 / np.max(ss3)
    su1 = np.concatenate((su[:50], su[100:]))
    suu = np.clip(su - np.mean(su1) + 1*np.mean(du1), 0 ,1) +.01
    du = medfilt(du, kernel_size = 5)
    diff = np.clip((du/np.max(du)  - suu/np.max(suu)), 0,1)
    diff/= np.max(diff)
    
    
    
    ss = bi*(n_f/s_f)*south
    ss = ss.T.flatten()
    sss = ss[~np.isnan(ss)] 
    nn = bi* north
    nn = nn.T.flatten()
    nnn = nn[~np.isnan(nn)]
    tt = np.clip(nnn/ np.max(nnn) - sss/np.max(sss), 0 ,1)

    # plt.figure()
    # plt.plot(du/ np.max(du) +3)
    # plt.plot(suu/ np.max(suu) + 3)
    # plt.plot(diff/ np.max(diff) +1)
    # plt.plot(nnn/np.max(nnn)+2)
    # plt.plot(sss/np.max(sss) +2)
    # plt.plot(tt/ np.max(tt))
    # plt.show()    
    
    


    from scipy.ndimage import gaussian_laplace
    from scipy.optimize import minimize

    x = np.arange(len(du))
    indices = np.concatenate((np.arange(35), np.arange(85, len(du))))
    indices = np.copy(x)

    not_needed = x[np.where((du - np.min(du)< .025 ))]
    # breakpoint()

    if len(not_needed)> 0:
        indices = indices[~np.isin(indices, not_needed)]
        
    sig = 5
    second_der = gaussian_laplace(du, sigma=sig)/ np.max(np.abs(gaussian_laplace(du, sigma=sig)))

    zrs = np.where(np.diff(np.sign(du* -second_der)) != 0)[0]

    def alignment_error(params, smooth_line, noisy_line, x_subset):
        # Unpack the parameters
        a00, a01, a02, a10, a11, a12 = params
        # Define the transformation matrix A
        A = np.array([[a00, a01, a02],
                      [a10, a11, a12],
                       [0, 0, 1]])
        
        # Construct the north_vector and opt_vector
        north_vector = np.vstack((x_subset, noisy_line[x_subset], np.ones_like(x_subset))).T
        opt_vector = np.dot(A, np.vstack((x_subset, smooth_line[x_subset], np.ones_like(x_subset))))
        # Calculate mean squared error
        mse = np.mean(np.sum((north_vector - opt_vector.T) ** 2, axis=1)) 
        return mse


    x0 = [1,0, 0, 0, 0, 0]

    result = minimize(alignment_error, x0, args=(suu, du, indices))
    a00, a01, a02, a10, a11, a12 = result.x
    A = np.array([[a00, a01, a02],
                  [a10, a11, a12],
                   [0, 0, 1]])
    aligned_smooth_line = np.dot(A, np.vstack((x, suu, np.ones_like(x))))[1]
    # aligned_smooth_line[35:80] = suu[35:80]
    aligned_smooth_line[not_needed] = suu[not_needed]
    

    bb = np.copy(suu)
    result = np.clip(du - bb, 0, 1 )/ np.max(np.clip(du - bb, 0, 1 ))
    
    peaks = scipy.signal.find_peaks(medfilt(du, 5),  distance = 3 )    
    plt.figure(figsize=(10, 5))
    plt.plot(x, du, label="Noisy Line", color="red")
    plt.plot(x, suu, label="Original Smooth Line", color="blue")
    # plt.plot(x, aligned_smooth_line, label="Aligned Smooth Line", color="green", linestyle="--")
    plt.scatter(peaks[0], du[peaks[0]], color = 'red')

    # plt.axvline(x = np.min(indices), color = 'red', label = 'axvline - full height')
    # plt.axvline(x = np.max(indices), color = 'red', label = 'axvline - full height')
    plt.legend()
    plt.show()

    # signal = (du* -second_der)
    # signal[:50] = 0
    # signal[125:] = 0 
    # plt.figure()
    # plt.plot(signal/ np.max(signal))
    # plt.scatter(zrs, (signal)[zrs])
    # plt.plot(du)
    # plt.plot(not_needed, du[not_needed] +.1)
    # plt.show()  
    # peaks = scipy.signal.find_peaks(signal, prominence = 0.1, distance = 5 )    
    # o_peaks = scipy.signal.find_peaks(-second_der, prominence = 0.2, distance = 5 )[0]

    # right_peak = peaks[0][np.max(np.where(peaks[0]< 125))]
    # left_peak = peaks[0][np.min(np.where(peaks[0]> 50))]
    # o_peaks = o_peaks[np.logical_and(o_peaks > left_peak, o_peaks < right_peak)]
    # valid_range = zrs[np.where((zrs > left_peak) & (zrs < right_peak))]
    # grad_array = np.diff(second_der)
    # try:
    #     min_v = np.min(valid_range[np.where(grad_array[valid_range] <0)]) 
    # except ValueError:
    #     breakpoint()
    
    # max_v = np.max(valid_range[np.where(grad_array[valid_range] >0)]) + 1 
    # if min_v > np.min(o_peaks):
    #     min_v = left_peak + int((np.min(o_peaks) - left_peak) /2) 
    # if max_v < np.max(o_peaks):
    #     max_v = right_peak + int((np.max(o_peaks) - right_peak) /2) 
    
    # plt.figure()
    # plt.plot(signal)
    # plt.axvline(x = min_v, color = 'red', label = 'axvline - full height')
    # plt.axvline(x = max_v, color = 'red', label = 'axvline - full height')
    # plt.show()  
    # breakpoint()
    return difference, plot_diff, result

