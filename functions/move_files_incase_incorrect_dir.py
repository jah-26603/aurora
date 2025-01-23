# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:22:01 2024

@author: JDawg
"""

import os
import shutil

cwd = r'D:\gold_level1c_2020_every_7th'

# Get list of all subdirectories within the base directory
folder_list = [x[0] for x in os.walk(cwd)][1:]  # Skip the root directory

for folder in folder_list:
    # Define the target 'data' subdirectory
    data_dir = os.path.join(folder, 'data')
    os.makedirs(data_dir, exist_ok=True)  # Create 'data' directory if not exists
    
    # Move all files from the current folder to the 'data' folder
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):  # Check if it's a file
            shutil.move(file_path, os.path.join(data_dir, file_name))
