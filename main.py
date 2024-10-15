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


north_day_base = "C:\\Users\\JDawg\\Desktop\\Aurora_Dates\\2020"
south_day_base = "C:\\Users\\JDawg\\Desktop\\Aurora_Dates\\2020"
# north_day_base = r"C:\Users\dogbl\Downloads\Aurora_Dates\2020"
# south_day_base = r'C:\Users\dogbl\Downloads\Aurora_Dates\2020'
north_days = [21, 51, 82, 112, 142, 172, 202, 234, 265, 295, 326, 356]
south_days = [202, 234, 82, 295, 326, 356, 21, 51, 265, 112, 142, 172]
# north_days = [308]
# south_days = [125]
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

    # process_loop(north_day, south_day, north_filepath, south_filepath, graphic_outputs)
    functions.create_gif(output_loc = f'{north_day_base}\\gifs', day = str(north_days[i]), **graphic_outputs, duration = 1000 )
