# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:13:03 2024

@author: JDawg
"""


import numpy as np
import glob

def time_window(filepath, antipode_filepath, time_window = (-20,20)):

    
    file_list = [name.replace(filepath,"") for name in glob.glob(rf'{filepath}/*.nc')]
    dict_keys = [file[27:32] for file in file_list]
    dict_scans = {key: [] for key in dict_keys}
    
    
    def time_to_minutes(time_str):
        hours, minutes = map(int, time_str.split('_'))
        return hours * 60 + minutes
    
    antipode_file_list = np.array([name.replace(antipode_filepath,"") for name in glob.glob(rf'{antipode_filepath}/*.nc')])
    afl_times_list = np.array([file[27:32] for file in antipode_file_list])
    a_int_keys = np.array([time_to_minutes(key) for key in afl_times_list])
    antipode_file_list = np.array([name for name in glob.glob(rf'{antipode_filepath}/*.nc')])

    
    int_keys = np.array([time_to_minutes(key) for key in dict_scans.keys()])
    
    for i, key in enumerate(dict_scans.keys()):
        result = -int_keys[i] + a_int_keys - time_window[0]
        idxs =  np.where(np.logical_and(result >= 0 , np.abs(result) < (time_window[1] - time_window[0])))
        dict_scans[key] = antipode_file_list[idxs]
        
        
    return dict_scans



if __name__ == "__main__":
    filepath = r'C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\Aurora_Dates\2020_11_19'
    antipode_filepath = r'C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\Aurora_Dates\2020_05_21'
    
    
    dict_scans = time_window(filepath, antipode_filepath, time_window = ((-20,20)))