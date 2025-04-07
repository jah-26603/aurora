# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:11:02 2025

@author: JDawg
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import cv2
from scipy.signal import medfilt
import cartopy.crs as ccrs
import yaml
import os
import pickle
import pandas as pd

fp = r'C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\products'
pkl_files = [f for f in os.listdir(fp) if f.endswith(".pkl")]

info_dict = {}
for i, file in enumerate(pkl_files):
    with open(f"{fp}/{file}", "rb") as f:
        info_dict[i] = pickle.load(f)

slices_dict = {}
count = 0    

for i, total_day_dict in enumerate(info_dict.values()):  # per day
    for j, day_scan_dict in enumerate(total_day_dict.values()):  # per scan in day
        
        if count == 0:
            start = int(np.nanmin(day_scan_dict['slice_map']))
            end = int(np.nanmax(day_scan_dict['slice_map']))
            slice_keys = [f'slice_{i}' for i in range(start, end + 1)]
            count += 1

        for lon_slice in day_scan_dict['slice_data']:  # per longitude slice
            
            if not lon_slice['species']:  # skip empty
                continue

            slice_id = lon_slice['slice_id']
            species_dict = lon_slice['species']
            
            for spec_key, spec_slice in species_dict.items():
                # initialize if first time seeing this species
                if spec_key not in slices_dict:
                    slices_dict[spec_key] = {k: [] for k in slice_keys}


                data = [spec_slice['ewb_lon'].item(), spec_slice['ewb_LT'].item(), spec_slice['ewb'],
                        spec_slice['pwb_lon'].item(), spec_slice['pwb_LT'].item(), spec_slice['pwb'],
                        spec_slice['peak_brightness'], spec_slice['peak_lat'][0], spec_slice['total_brightness']]
                
                slices_dict[spec_key][f'slice_{slice_id}'].append(data)



columns = [
    'ewb_lon', 'ewb_LT', 'ewb_lat',
    'pwb_lon', 'pwb_LT', 'pwb_lat',
    'peak_brightness', 'peak_lat', 'total_brightness'
]



for spec_key in slices_dict:
    for slice_key in slices_dict[spec_key]:
        data = slices_dict[spec_key][slice_key]
        if isinstance(data, list) and data:  # check it's a list and non-empty
            slices_dict[spec_key][slice_key] = pd.DataFrame(data, columns=columns)

import matplotlib.pyplot as plt
import numpy as np

for spec_key, slice_data in slices_dict.items():
    for slice_id, df in slice_data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        # Determine marker properties
        equal_mask = df['ewb_lat'] == df['pwb_lat']
        not_equal_mask = ~equal_mask

        # # Plot
        plt.figure(figsize=(6, 4))

        # Black circle for equal values
        plt.scatter(df.loc[equal_mask, 'ewb_LT'], df.loc[equal_mask, 'ewb_lat'], 
                    color='black', edgecolors='black', label='Equal', marker='o')

        # Blue ring for EWB (only when different)
        plt.scatter(df.loc[not_equal_mask, 'ewb_LT'], df.loc[not_equal_mask, 'ewb_lat'], 
                    facecolors='none', edgecolors='blue', label='EWB ', marker='o')

        # Green ring for PWB (only when different)
        plt.scatter(df.loc[not_equal_mask, 'pwb_LT'], df.loc[not_equal_mask, 'pwb_lat'], 
                    facecolors='none', edgecolors='green', label='PWB ', marker='o')

        # Formatting
        plt.xlabel('Local Time')
        plt.ylabel('Latitude')
        plt.xlim(0, 24)
        plt.ylim(50, 80)
        plt.title(f"{spec_key} - [{round(df['ewb_lon'].min(), 2)}, {round(df['ewb_lon'].max(), 2)}] LONG ({slice_id})")
        plt.legend()
        plt.tight_layout()
        plt.show()

        
        
        # fig, ax1 = plt.subplots(figsize=(6, 4))

        # # Left axis: Peak Brightness vs Local Time
        # ax1.scatter(df['ewb_LT'], df['peak_brightness'], color='red', label='Peak Brightness')
        # ax1.set_xlabel('Local Time')
        # ax1.set_ylabel('Peak Brightness (R)', color='red')
        # ax1.tick_params(axis='y', labelcolor='red')
        # ax1.set_xlim(0, 24)

        # # Title
        # ax1.set_title(f'{spec_key} - [{round(df["ewb_lon"].min(),2)}, {round(df["ewb_lon"].max(),2)}] LONG ({slice_id})')

        # # Right axis: Peak Latitude
        # ax2 = ax1.twinx()
        # ax2.scatter(df['ewb_LT'], df['peak_lat'], color='black', marker='o', label='Peak Brightness Latitude')
        # ax2.set_ylabel('Latitude (°)', color='black')
        # ax2.tick_params(axis='y', labelcolor='black')
        # ax2.set_ylim(50, 80)

        # # Legends
        # ax1.legend(loc='upper left')
        # ax2.legend(loc='upper right')

        # plt.tight_layout()
        # plt.show()



# import matplotlib.pyplot as plt

# for spec_key, slice_data in slices_dict.items():
#     for slice_id, df in slice_data.items():
#         if not isinstance(df, pd.DataFrame) or df.empty:
#             continue

#         # Plot 1: EWB Latitude vs Local Time
#         plt.figure(figsize=(6, 4))
#         plt.scatter(df['ewb_LT'], df['ewb_lat'], color='blue', label='EWB')
#         plt.xlabel('Local Time')
#         plt.xlim(0,24)
#         plt.ylim(np.min(df['ewb_lat']) - 5, np.max(df['ewb_lat']) + 5 )
#         plt.ylabel('EWB Latitude')
#         plt.title(f'{spec_key} - [{round(np.min(df['ewb_lon']),2)}, {round(np.max(df['ewb_lon']),2)}] LONG ({slice_id}) ')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#         # Plot 2: PWB Latitude vs Local Time
#         plt.figure(figsize=(6, 4))
#         plt.scatter(df['pwb_LT'], df['pwb_lat'], color='green', label='PWB')
#         plt.xlabel('Local Time')
#         plt.ylabel('PWB Latitude')
#         plt.xlim(0,24)
#         plt.ylim(np.min(df['pwb_lat']) - 5, np.max(df['pwb_lat']) + 5 )
#         plt.title(f'{spec_key} -  [{round(np.min(df['pwb_lon']),2)}, {round(np.max(df['pwb_lon']),2)}] LONG ({slice_id})')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
# count = 0    
# for i in range(len(list(info_dict.keys()))): #per day
#     total_day_dict = info_dict[i]
    
#     for j in range(len(list(total_day_dict.keys()))): # per scan in day
#         day_scan_dict = total_day_dict[j]
        
        
#         if count == 0:
#             start = int(np.nanmin(day_scan_dict['slice_map']))
#             end = int(np.nanmax(day_scan_dict['slice_map']))
#             pwb_dict = {f'slice_{i}': [] for i in range(start, end+1)}   
#             ewb_dict = {f'slice_{i}': [] for i in range(start, end+1)}  
        
#         for k in range(len(day_scan_dict['slice_data'])): # per longitude slice
            
#             lon_slice = day_scan_dict['slice_data'][k]
            
#             if len(lon_slice['species']) == 0: #empty set
#                 continue
#             slice_id = lon_slice['slice_id']
#             try:
#                 spec_slice = lon_slice['species']['1356']
#             except KeyError:
#                 continue
#             dummy1 = [spec_slice['ewb_lon'].item(), spec_slice['ewb_LT'].item(), spec_slice['ewb']]
#             dummy2 = [spec_slice['pwb_lon'].item(), spec_slice['pwb_LT'].item(), spec_slice['pwb']]
#             pwb_dict[f'slice_{slice_id}'].append(dummy1)
#             ewb_dict[f'slice_{slice_id}'].append(dummy2)
            
            