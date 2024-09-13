# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:13:03 2024

@author: JDawg
"""

import functions
import numpy as np



filepath = r'C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\Aurora_Dates\03_19_2020_wk'
dict_scans = functions.scans_at_time(filepath)
time_window = 30 # minutes on both sides


def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split('_'))
    return hours * 60 + minutes

int_keys = [time_to_minutes(key) for key in dict_scans.keys()] 



def idx_to_key(idxs):
    keys_in_window =[]
    for i in idxs:
        time = int_keys[i]
        hour = str(time//60).zfill(2)
        minute = str(time % 60).zfill(2)
        string = hour+'_'+minute
        keys_in_window.append(string)
    return keys_in_window
        
        
for key in int_keys:
    current_window_indices = np.where(np.abs(np.array(int_keys) - key) < time_window)[0]
    keys_in_window = idx_to_key(current_window_indices)
    