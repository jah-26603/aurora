# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:19:55 2025

@author: JDawg
"""

import numpy as np
import netCDF4 as nc
import yaml
from tqdm import tqdm
import functions  # Assuming all helper functions are in this module
import matplotlib.pyplot as plt
class SpeciesProcessor:
    def __init__(self, specie, species_info_dict, file_idx, time, radiance, latitude, wavelength, dict_list_south_scans, keys):
        self.specie = specie
        self.specie_info = species_info_dict[specie]
        self.file_idx = file_idx
        self.radiance = radiance
        self.latitude = latitude
        self.wavelength = wavelength
        self.dict_list_south_scans = dict_list_south_scans
        self.keys = keys
        self.time = time
        # Initialize arrays for storing results
        self.brightnesses = None
        self.south = None
        self.difference = None
        self.time_of_scan = None


    def process_species(self, sza, ema, count):
        # Process the species for this file
        print(count)
        # if count> 1:
        #     breakpoint()
        
        # Get indices and time
        filled_indices, one_pixel = functions.filled_indices(self.wavelength)  # acceptable indices for analysis
        date, time_array = functions.date_and_time(filled_indices, self.time)  # gets date and time
        self.time_of_scan = time_array[91, 81]

        # Raw Data Arrays
        self.brightnesses = functions.get_data_info(self.radiance, one_pixel, **self.specie_info)

        self.south = functions.get_south_half(self.dict_list_south_scans[self.keys[self.file_idx]], self.brightnesses, self.specie_info)
        # Apply background mask
        background_mask = np.where(np.isnan(self.latitude), 0, 1)
        self.brightnesses = (self.brightnesses * background_mask)[52:]
        self.south = (self.south * background_mask)[:52]

        try:
            self.difference, diff = functions.absolute_difference(self.brightnesses, self.south, np.copy(self.latitude), sza, ema, count)
        except ValueError:
            print("Missing result from", self.time_of_scan)
            return None  # Return None to indicate that this species could not be processed

        if np.isnan(self.difference).all():
            print('Missing result from', self.time_of_scan)
            return None
        # Find the border and calculate additional quantities
        border_image, lb, rb, dummy_diff, dummy_second_der, dminv, dmaxv = functions.find_edge(self.difference, diff, self.latitude, sza)
        # # Return results in a dictionary
        return {
            'points': [lb, rb],
            'opoints': [dminv, dmaxv],
            'diff': dummy_diff,
            'second_der': dummy_second_der,
            'time_of_scan': self.time_of_scan,
            'difference': self.difference,
            'brightnesses': self.brightnesses,
            'south': self.south
        }
