# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 18:11:52 2025

@author: JDawg
"""


import functions 
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import cv2
from scipy.signal import medfilt
import cartopy.crs as ccrs
import yaml
import functions.SpeciesProcessor as SP
import os
import apexpy

# Calculates Universal Time
def extract_utc(byte_chars):
    """Convert byte characters to a string and extract HH:MM:SS.sss, then convert to Julian time."""
    try:
        time_str = b''.join(byte_chars).decode()  # Convert byte array to string
        if 'T' not in time_str:  # Check if format is correct
            return np.nan
        time_part = time_str.split('T')[1].rstrip('Z')  # Extract time part (HH:MM:SS.sss)
        
        # Split the time into hours, minutes, and seconds
        time_parts = time_part.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        
        # Convert to Julian time (decimal hours)
        julian_time = hours + minutes / 60 + seconds / 3600
        return julian_time
    except Exception:
        return np.nan  # Return NaN in case of an error (e.g., invalid format)

def orange_slices(lon, slice_width = 7):
    bin_edges = np.arange(np.nanmin(lon), np.nanmax(lon) + slice_width, slice_width)
    slice_map = np.full(lon.shape, np.nan)
    valid_mask = ~np.isnan(lon)
    slice_map[valid_mask] = np.digitize(lon[valid_mask], bin_edges) - 1 #slices
    return slice_map


def aurora_products(lat, lon, time, results, result_masks):
    # Magnetic Coordinates conversion
    apex = apexpy.Apex()
    mag_lat, mag_lon = apex.convert(lat, lon, 'geo', 'qd')
    
    # Universal Time (UTC) extraction and conversion
    vectorized_extract_utc = np.vectorize(extract_utc, signature="(n)->()")
    UTC = vectorized_extract_utc(time)
    LT = (UTC + lon / 15) % 24
    MLT = apex.mlon2mlt(mag_lon, UTC)  # Magnetic Local Time
    
    # Orange Slice Map
    slice_map = orange_slices(lon, slice_width=10)[52:]
    mag_slice = orange_slices(mag_lon, slice_width=10)[52:]
    
    # Compute auroral properties
    num_auroral_pixels = np.count_nonzero(result_masks.sum(axis=0) >= 2)
    O_LBH_ratio = np.nan_to_num(results['1356'] / results['LBH'], nan=0.0, posinf=0.0, neginf=0.0)
    O_N_ratio = np.nan_to_num(results['1356'] / results['1493'], nan=0.0, posinf=0.0, neginf=0.0)
    LBH_N_ratio = np.nan_to_num(results['LBH'] / results['1493'], nan=0.0, posinf=0.0, neginf=0.0)
    
    lat = lat[52:]
    lon = lon[52:]
    LT = LT[52:]
    species_dict = results
    
    slice_data = []  # List to store per-slice properties

    for i in range(int(np.nanmax(slice_map)) + 1):
        pxs = np.where((slice_map == i))
        slice_info = {"slice_id": i, "species": {}}

        for name, specie in species_dict.items():
            in_slice = specie[pxs]
            in_lat = lat[pxs]
            in_lon = lon[pxs]
            in_LT = LT[pxs]
            if np.all(in_slice == 0):
                continue

            masked_slice = np.where(in_slice == 0, np.nan, in_slice)  # Replace zeros with NaN
            peak_brightness = np.nanmax(in_slice)
            peak_lat = in_lat[np.where(in_slice == peak_brightness)]
            pwb = np.nanmax(in_lat[np.where(~np.isnan(masked_slice))])
            ewb = np.nanmin(in_lat[np.where(~np.isnan(masked_slice))])
            
            pwb_lon = in_lon[in_lat == pwb].data
            ewb_lon = in_lon[in_lat == ewb].data
            pwb_LT = in_LT[in_lat == pwb].data
            ewb_LT = in_LT[in_lat == ewb].data
            total_brightness = np.nansum(in_slice)

            # Store computed values for this species
            slice_info["species"][name] = {
                "peak_brightness": peak_brightness,
                "peak_lat": peak_lat.tolist(),  # Convert to list for serialization
                "pwb": pwb,
                "ewb": ewb,
                "pwb_lon": pwb_lon,
                "ewb_lon": ewb_lon,
                "pwb_LT": pwb_LT,
                "ewb_LT": ewb_LT,
                "total_brightness": total_brightness
            }
        # slice_info["middle_lon"] = np.median(lon[pxs])
        slice_data.append(slice_info)

    # Store all computed results in a dictionary
    products = {
        "mag_lat": mag_lat,
        "mag_lon": mag_lon,
        "UTC": UTC,
        "LT": LT,
        "MLT": MLT,
        "slice_map": slice_map,
        "mag_slice_map": mag_slice,
        "num_auroral_pixels": num_auroral_pixels,
        "O_LBH_ratio": O_LBH_ratio,
        "O_N_ratio": O_N_ratio,
        "LBH_N_ratio": LBH_N_ratio,
        "slice_data": slice_data  # Storing per-slice calculations
    }
    return products


        
    # # Visualization for slices
    # plt.figure()
    # plt.imshow(slice_map, cmap='viridis', interpolation='none')
    # plt.colorbar(label="Region Label")
    # plt.xlabel("X Index")
    # plt.ylabel("Y Index")
    # plt.title("lon Regions (NaNs Unclassified)")
    # plt.show()