# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:53:15 2025

@author: dogbl
"""

import pandas as pd
from datetime import datetime
import numpy as np

get_south = True
num_days = 365
save_days_list = True


def to_julian_day(year, month, day):
    dt = datetime(int(year), int(month), int(day))
    return dt.timetuple().tm_yday  # Extracts the Julian day

def kp_days(lower_bound = 0, upper_bound = 3, year = 2020):
    
    fp = r"C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\get_files_download\2019_2024_KP_INDICES.txt"
    df = pd.read_csv(fp, delim_whitespace=True)
    df['julian_day'] = df.apply(lambda row: to_julian_day(2020, row['m'], row['dy']), axis=1)
    
    
    df_max_kp = df.loc[df.groupby(['year', 'm', 'dy'])['kp'].idxmax()]
    days = df_max_kp[(df_max_kp.kp >= lower_bound) & (df_max_kp.kp <= upper_bound)]
    days = days[days.year == year]
    
    acceptable_south_days = days[['julian_day', 'kp']]
    all_days = np.arange(1,366 +1)
    difference_list = []
    
    for i in range(len(all_days)):
        dummy = np.abs(all_days[i] - np.array(acceptable_south_days['julian_day']))
        difference_list.append(np.min(dummy))
    
    difference_list = np.array(difference_list)
    
    if get_south:
        print('Number of days with southern scan > 6 days:', len(difference_list[difference_list >= 7]))
        print('Largest day gap for no available southern scans:', np.max(difference_list))
    if save_days_list:
        selected_indices = np.linspace(0, len(acceptable_south_days) - 1, num_days, dtype=int)
        selected_rows = acceptable_south_days.iloc[selected_indices]
        final = selected_rows['julian_day'].astype(int).astype(str).str.zfill(3)
        ##download every day
        # final = np.char.zfill(all_days.astype(str), 3)  # Correct way to zero-pad in NumPy    
        # np.savetxt("2020_days.txt", final, fmt="%s")
        # # Save to file
        final.to_csv(f'{year}_days.txt', index=False, header=False)
        
    return days.julian_day