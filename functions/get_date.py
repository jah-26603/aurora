# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:59:47 2024

@author: JDawg
"""
import numpy as np

def date_and_time(filled_indices, time):
    #Extract date and times
    date = (''.join([c.decode('utf-8') for c in time[filled_indices[0,0],filled_indices[0,1]]]))[:10]
    time_array = np.array([[None]*92]*104)
    for index in filled_indices:
        i, j = index
        time_array[i,j] = (''.join([c.decode('utf-8') for c in time[i, j]]))[11:]
    
    
    return date, time_array