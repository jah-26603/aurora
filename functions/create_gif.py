# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 23:45:48 2024

@author: JDawg
"""
import imageio
from PIL import Image
import glob
import os 
def create_gif(output_loc, day, raw_north, difference, results, duration = 100):
    
    
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
        
        
    
    # Load the raw PNG images from the specified directory
    png2 = [Image.open(file) for file in glob.glob(f'{difference}/*.png')]  # Adjust extension to '.png'
    png3 = [Image.open(file) for file in glob.glob(f'{results}/*.png')]  # Adjust extension to '.png'
    png1 = [Image.open(file) for file in glob.glob(f'{raw_north}/*.png')]  # Adjust extension to '.png'
    
    # Ensure all lists have the same number of frames
    num_frames = min(len(png1), len(png2), len(png3))

    # Assuming all PNGs have the same size
    width, height = png1[0].size  # Get dimensions from the first PNG

    # Create a list to hold the combined frames
    combined_frames = []

    # Iterate through the frames of the PNG sets
    for i in range(num_frames):
        # Create a blank canvas for the combined image
        combined_frame = Image.new('RGB', (width * 3, height))  # 3 images side by side

        # Paste each PNG's frame onto the combined frame
        combined_frame.paste(png1[i], (0, 0))        # First image
        combined_frame.paste(png2[i], (width, 0))    # Second image
        combined_frame.paste(png3[i], (width * 2, 0)) # Third image

        # Append the combined frame to the list
        combined_frames.append(combined_frame)

    # Save the combined GIF
    imageio.mimsave(f'{output_loc}\\results_{day}.gif', combined_frames, duration= duration, loop = 0)  # Adjust duration as needed