# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:17:32 2024

@author: JDawg
"""
from smoothlbh_unsupervised import process_loop
from tqdm import tqdm
import os 
import functions
import warnings




warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

start_file, minus_end_file = (0, 0)



south_day_base = r'D:\gold_level1c_2020_every_7th'
north_day_base = r'D:\gold_level1c_2020_every_7th'
north_days, south_days = functions.get_south_days(south_day_base, north_day_base)

param = ['LBH', '1493', '1356']
species_info_fp = r"C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\config.yaml"
# qq = ['LBH']
# qq = ['1356']
# qq = ['1493']

for i in tqdm(range(0, len(north_days))):

        north_day = f"{north_day_base}\\{north_days[i]:03d}"
        south_day = f"{south_day_base}\\{south_days[i]:03d}"
        north_filepath = f"{north_day}\\data"
        south_filepath = f"{south_day}\\data"
        
        graphic_outputs = {
            'raw_north': f"{north_day}\\graphics\\raw_north",
            'difference': f"{north_day}\\graphics\\difference",
            'results': f"{north_day}\\graphics\\results"
        }

        process_loop(param, start_file, minus_end_file, north_day, south_day, 
                     north_filepath, south_filepath, graphic_outputs, species_info_fp)
        
        # Create GIF
        # functions.create_gif(output_loc=f'{north_day_base}\\gifs',
        # day=str(north_days[i]),
        #     **g_out_lbh,
        #     duration=1000,
        #     species=spec
        # )
