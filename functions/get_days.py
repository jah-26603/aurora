# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:28:21 2024

@author: JDawg
"""
import os
import numpy as np

def get_south_days(s_fp, n_fp):
    
    
    dummy = next(os.walk(s_fp))[1]
    south_days = np.array([int(ii) for ii in dummy])
    south_days = (south_days + 183) % 365
    dummy = next(os.walk(n_fp))[1]
    north_days = np.array([int(ii) for ii in dummy])

    south_days_list = np.array([north_days[np.argmin(np.abs(north_days - south_day))]
    for south_day in south_days])

    return north_days, south_days_list



