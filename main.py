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
import winsound
import subprocess
import traceback

     

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

start_file, minus_end_file = (0, 0)

south_day_base = r'E:\gold_level1c_2020'
north_day_base = r'E:\gold_level1c_2020'

north_days, south_days = functions.get_south_days(south_day_base, north_day_base, kp = True, kp_lb = 4, kp_up = 6, year =2020)

param = ['LBH', '1493', '1356']
species_info_fp = r"C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\config.yaml"

try:
    
    for i in tqdm(range(23, 25)):
    
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
            
            
    subprocess.run(["python", r"create_ppt_presentation.py"])
            # Create GIF
            # functions.create_gif(output_loc=f'{north_day_base}\\gifs',
            # day=str(north_days[i]),
            #     **g_out_lbh,
            #     duration=1000,
            #     species=spec
            # )

except Exception as e:
    winsound.Beep(200, 100)    # Frequency = 200 Hz, Duration = 100 ms
    tb = traceback.format_exception(e.__class__, e, e.__traceback__)
    print("".join(tb))  # Prints the complete traceback