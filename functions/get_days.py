# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:28:21 2024

@author: JDawg
"""
import os
import numpy as np
import functions
def get_south_days(s_fp, n_fp, kp = True, kp_lb = 0, kp_up = 3, year = 2020):
    
    if kp:
        south_days = np.array(functions.kp_days(lower_bound = 0, upper_bound = 3, year = 2020)) #this is going to be a constant for now
        north_days = np.array(functions.kp_days(lower_bound = kp_lb, upper_bound = kp_up, year = year))
        
    else:
        dummy = next(os.walk(s_fp))[1]
        south_days = np.array([int(ii) for ii in dummy])
        dummy = next(os.walk(n_fp))[1]
        north_days = np.array([int(ii) for ii in dummy])


    south_days_list = np.array([south_days[np.argmin(np.abs((south_days - north_day +182)%365))]
    for north_day in north_days])
    
    
    return north_days, south_days_list



