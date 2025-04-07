# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:23:06 2025

@author: JDawg
"""
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import cv2
import os
import functions


save_dir = r'E:\gold_level1c_2019_saved_arrays'
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
path = r'E:\gold_level1c_2019'
os.chdir(path)
all_day_folders = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
count = 0 
LBH = {
  'lower_bound': 138,
  'upper_bound': 152,
  'lower_bound_2': 148,
  'upper_bound_2': 150,
  'multi_regions': True}

def limb_data (arr, bi):
    k0 = (arr[:52]*bi).T
    k1 = k0.flatten()
    k2 = k1[~np.isnan(k1)]
    return k2


for i in tqdm(range(len(all_day_folders))):
    
    file_list = glob.glob(f'{all_day_folders[i]}/data/*.nc')
    if len(file_list) == 0:
        file_list = glob.glob(f'{all_day_folders[i]}/*.nc')
        
    south_scans, sza_scans, ema_scans = [], [], []

    
    for file in tqdm(range(0, len(file_list))):
        try:
            ds = nc.Dataset(file_list[file], 'r')
        except OSError:
            print('Error in reading netcdf4 file')
            continue

        #loads data
        radiance = ds.variables['RADIANCE'][:]
        wavelength = ds.variables['WAVELENGTH'][:]
        sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
        ema = ds.variables['RAY_SOLAR_PHASE_ANGLE'][:]
        #skips all northern scans
        if np.isnan(sza[0].data).all():
            continue
        
        radiance = np.clip(radiance, 0, np.inf)
        day = ds.DATE_START[:10]
        
        if count == 0:
            count += 1
            
            
            latitude = ds.variables['GRID_LAT'][:]               
            dummy = np.where(np.isnan(latitude), 0, 1)
            kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])
    
            nb = cv2.filter2D(dummy.astype(np.float32), -1, kernel)
            bi = np.where((nb < 7) & (dummy == 1), 1, np.nan)[:52]
            
            
        filled_indices, one_pixel = functions.filled_indices(wavelength) 
        brightnesses = functions.get_data_info(radiance, one_pixel, **LBH)
        
        south_scans.append(limb_data(brightnesses[:52], bi).compressed())
        sza_scans.append(limb_data(sza[:52], bi).compressed())
        ema_scans.append(limb_data(ema[:52], bi).compressed())

        # Convert lists to NumPy arrays
    south_scans = np.array(south_scans, dtype=object)
    sza_scans = np.array(sza_scans, dtype=object)
    ema_scans = np.array(ema_scans, dtype=object)
 
    # Save as .npy files
    np.save(os.path.join(save_dir, f"south_scans_{day}.npy"), south_scans)
    np.save(os.path.join(save_dir, f"sza_scans_{day}.npy"), sza_scans)
    np.save(os.path.join(save_dir, f"ema_scans_{day}.npy"), ema_scans)
    
    print(f"Saved: south_scans_{day}.npy, sza_scans_{day}.npy, ema_scans_{day}.npy")
    
    


