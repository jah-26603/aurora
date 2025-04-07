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
from scipy.signal import medfilt as medfilt
from sklearn.decomposition import FastICA
# from train_limb import SimpleNN
import torch.nn as nn
import torch


def limb_data (arr, bi):
    k0 = (arr[52:]*bi).T
    k1 = k0.flatten()
    k2 = k1[~np.isnan(k1)]
    return k2

def absolute_difference(north, south, latitude, sza, ema, count):
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
    du = medfilt(du, kernel_size = sig)
    result = np.clip(du - suu, 0, 1 )/ np.max(np.clip(du - suu, 0, 1 ))
    


    class SimpleNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                # Adding 3 more layers:
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )
    
    
        
        def forward(self, x):
            return self.model(x)
    # Prepare input data
    test_sza = limb_data(np.copy(sza), bi)  # Shape (124,)
    test_ema = limb_data(np.copy(ema), bi)
    
    if len(test_sza) != 124:
        breakpoint()

    new_X_tensor = torch.tensor(test_sza, dtype=torch.float32).unsqueeze(0)
    # new_X_tensor = torch.tensor(np.column_stack((test_sza, test_ema)).flatten(), dtype=torch.float32).unsqueeze(0)
    # Load model
    model = SimpleNN(124, 124)
    model.load_state_dict(torch.load(
        r"C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\Empirical_southern_limb\scan_predictor_model_2019_2020.pth",
        weights_only=True  # Avoid potential security risks
    ))
    
    
    model.eval()
    # Run inference
    with torch.no_grad():
        predicted_south_scan = model(new_X_tensor).cpu().numpy()
    # if count%2==0:
    ll = kk2/np.max(kk2)
    ll = medfilt(ll, kernel_size= 9)/ np.max(medfilt(ll, kernel_size= 9))
    nn_pred = medfilt(predicted_south_scan[0], kernel_size= 9)/ np.max(medfilt(predicted_south_scan[0], kernel_size= 9))
    reg_pred = medfilt(ss2, kernel_size= 9)/ np.max(medfilt(ss2, kernel_size= 9))
    
    aa = np.mean(np.concatenate((ll[:40], ll[90:])))
    bb = np.mean(np.concatenate((nn_pred[:40], nn_pred[90:])))
    cc = np.mean(np.concatenate((reg_pred[:40], reg_pred[90:])))
    

    
    
    # plt.figure()
    # plt.plot(nn_pred*aa/bb, label = 'Neural Network Estimation')
    # plt.plot(reg_pred*aa/cc, label = 'Raw South Brightness')
    # plt.plot(kk2/np.max(kk2), label = 'Raw North Brightness', linestyle='dashed')
    # # plt.plot(np.clip(kk2/np.max(kk2) - predicted_south_scan[0] + 1,1, ))
    # plt.legend()
    # plt.show()
    
    ll = kk2/np.max(kk2)
    ll = medfilt(ll, kernel_size= 5)/ np.max(medfilt(ll, kernel_size= 5))

    aabb = np.clip(ll-nn_pred*aa/bb, 0 , np.inf)
    aacc = np.clip(ll-reg_pred*aa/cc, 0 , np.inf)
    




    # Elementwise Minimum
    min_signal = np.minimum(aabb, aacc)
    
    # Averaging
    avg_signal = (aabb + aacc) / 2

    # plt.figure()
    # plt.plot(aabb / np.max(aabb), label='Signal 1 (Normalized)')
    # plt.plot(aacc / np.max(aacc) - 1, label='Signal 2 (Normalized)')
    # plt.plot(aabb * aacc / np.max(aabb * aacc) - 2, label='Elementwise Multiplication')
    # plt.plot(min_signal / np.max(min_signal) - 3, label='Elementwise Minimum')
    # plt.plot(avg_signal / np.max(avg_signal) - 4, label='Averaged Signal')

    # plt.legend()
    # plt.show()

    result = aabb/np.max(aabb)
    result = (aabb*aacc/(np.max(aacc*aabb)) + min_signal/ np.max(min_signal) + avg_signal/ np.max(avg_signal))
    result = np.copy(avg_signal)
    
    return difference, result

