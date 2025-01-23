# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:09:45 2024

@author: JDawg
"""
import functions 
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import cv2
from scipy.signal import medfilt
import cartopy.crs as ccrs
import time
import yaml

def process_loop(species, start_file, minus_end_file, north_day, south_day,
                 north_filepath, south_filepath, graphic_outputs, species_info_fp):
    file_list = glob.glob(f'{north_filepath}/*.nc')
    hemisphere_order = []
    lat_threshold = 50
    dict_list_south_scans = functions.time_window(north_filepath, south_filepath, time_window = (-20, 20))
    keys = list(dict_list_south_scans.keys())
    
    species_info_dict = yaml.safe_load(open(species_info_fp))
    count = 0
    

    points, opoints = (np.zeros((len(keys), 2)) for _ in range(2))
    difference_array, brightnesses_array = (np.zeros((len(keys), 52, 92)) for _ in range(2))
    diff_array, second_der_array, time_of_scan_array, day_array, file_arr = ([] for _ in range(5))

    for file in tqdm(range(start_file, int(len(keys))- minus_end_file)):
        

        try:
            ds = nc.Dataset(file_list[file], 'r')
        except OSError:
            print('Error in reading netcdf4 file')
            continue
        if count == 0:
            count += 1
            day = ds.DATE_START[:10]
            print('\n')
            print('Northern Hemisphere Scans from:   ' + day)

        
        latitude = ds.variables['GRID_LAT'][:]    #Note: latitude data is flipped!! So data at top is southern most latitude and data at bottom is Northern most latitude
        radiance = ds.variables['RADIANCE'][:]
        wavelength = ds.variables['WAVELENGTH'][:]
        radiance = np.clip(radiance, 0, np.inf)
        sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
        

        hemisphere_order, hemisphere, skip_s, skip_n = functions.hemisphere(hemisphere_order, sza, skip_s = True, skip_n = False) #which hemisphere  
        if skip_s == 1 and hemisphere_order[-1] == 1:
            continue
        if skip_n == 1 and hemisphere_order[-1] == 0:
            continue
        
        breakpoint()
        for specie in species:
            specie_info = species_info_dict[specie]
            
            filled_indices, one_pixel = functions.filled_indices(wavelength) #acceptable indices for analysis
            date, time_array =  functions.date_and_time(filled_indices, time) #gets date and time
            time_of_scan = time_array[91,81]
            
            #Raw Data Arrays
            brightnesses = functions.get_data_info(radiance, one_pixel, **specie_info)
            south = functions.get_south_half(dict_list_south_scans[keys[file]], brightnesses, specie_info)
        
            if hemisphere_order[-1] == 1:
                continue
        
       
            'Work in progress...'
            #This applies a background mask to get rid of stars so their intensities aren't considered
            background_mask = np.where(np.isnan(latitude), 0 ,1)
            brightnesses = (brightnesses*background_mask)[52:]
            south = (south*background_mask)[:52]
            
            try:
                difference, diff = functions.absolute_difference(brightnesses, south, np.copy(latitude))
            except ValueError:
                print("Missing result from", time_of_scan )
                continue
            if np.isnan(difference).all():
                print('Missing result from', time_of_scan)
                continue
            border_image, lb, rb, dummy_diff, dummy_second_der, dminv, dmaxv = functions.find_edge(difference, diff, latitude)
            points[file] = [lb, rb]
            opoints[file] = [dminv,dmaxv]
            diff_array.append(dummy_diff)
            second_der_array.append(dummy_second_der)
            time_of_scan_array.append(time_of_scan)
            difference_array[file] = difference
            day_array.append(day)
            brightnesses_array[file] = brightnesses
            file_arr.append(file)
            



    
    difference_array = np.array([arr for arr in difference_array if not np.all(arr == 0)])
    brightnesses_array = np.array([arr for arr in brightnesses_array if not np.all(arr == 0)])
    pp = points[~(points == 0).all(axis=1)]
    op = opoints[~(opoints == 0).all(axis=1)] #just for plotting
    smooth_points = np.column_stack((medfilt(pp[:, 0], 7), medfilt(pp[:, 1], 7)))
    actual_points = np.copy(pp)
    c, r = np.where(np.abs(smooth_points - pp) > 3)
    actual_points[c, r] = smooth_points[c, r]

    actual_points = np.copy(pp)
    plt.figure()
    plt.plot(medfilt(pp[:,1],7), color = 'black', label = 'Smooth Decision Boundaries')
    plt.plot(medfilt(pp[:,0],7), color = 'black')
    plt.plot(pp, color = 'red', label = 'OG Decision Boundaries')
    plt.title('Decision Boundary vs file number (time)')
    plt.ylabel('Column Boundaries for limb exclusion')
    plt.legend()
    plt.show()
    
    for i, file in enumerate(file_arr):
        dummy = np.copy(latitude)
        dummy[np.isnan(dummy)] = 0
        dummy[dummy != 0 ] = 1
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])

        neighbor_check = cv2.filter2D(dummy, -1, kernel)
        border_image = np.where((neighbor_check < 8) & (dummy == 1), 1, 0)       #perhaps change to 7?
        border_image = border_image[52:]
        border_image = np.abs(border_image - 1)
        border_image[:,int(pp[i,0]) :int(pp[i,1])] = 1
        functions.results_loop(time_of_scan_array[i], difference_array[i], border_image, 
                     keys, file_arr[i], latitude, lat_threshold, day_array[i],
                     brightnesses_array[i], actual_points[i], graphic_outputs, species)
    
    # for i in range(difference_array.shape[0]):
    #     fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # Adjust `figsize` as needed
    
    #     # Display the image on the first subplot
    #     axes[0].imshow(difference_array[i], aspect='auto')
    #     axes[0].axvline(x=pp[i, 0], color='white')
    #     axes[0].axvline(x=pp[i, 1], color='white')
    #     axes[0].axis('off')  
    
    #     # Plot the data on the second subplot
    #     axes[1].plot(second_der_array[i] + 2, label='Second Derivative')
    #     axes[1].plot(diff_array[i], label='Difference Array')
    #     axes[1].axvline(x=op[i, 0], color='red')
    #     axes[1].axvline(x=op[i, 1], color='red')
    #     axes[1].legend()
    #     axes[1].set_xlim([15, 124-16])  
    #     # axes[1].axis('off')
              
    #     # Adjust layout and show the figure
    #     plt.tight_layout()
    #     plt.show()
    
    
    
    
    
    
    
"""plotting stuff"""
#Zoomed on in auroral region 
# import cartopy.crs as ccrs

# # Setup figure with one subplot
# fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=longitude[0, 51])})

# # Add a title for the figure
# fig.suptitle(
#     f'Radiance Colormap Zoomed \nDate: {date}\nTime: {time_array[filled_indices[0, 0], filled_indices[0, 1]]}', 
#     color='black', y=1.02, weight='bold'
# )

# # Set the geographical extent of the plot
# ax.set_extent([-115, 15, 30, 80])

# # Add gridlines
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=2, color='gray', alpha=0.5, linestyle='--')

# # Plot the data
# latitude = latitude[52:]
# longitude = longitude[52:]
# im = ax.pcolor(longitude, latitude, brightnesses,  # Choose the first dataset for single image
#                transform=ccrs.PlateCarree(), cmap='plasma')

# # Add a color bar
# cb = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05)
# cb.set_label(f'{units[0]}')  # Use the cor
# plt.show()

if __name__ == "__main__":
    north_day =  "C:\\Users\\JDawg\\Desktop\\Aurora_Dates\\2020\\295"
    south_day = "C:\\Users\\JDawg\\Desktop\\Aurora_Dates\\2021\\112"
    
    north_filepath = f"{north_day}\\data"
    south_filepath = f"{south_day}\\data"
    
    graphic_outputs = {
        'raw_north': f"{north_day}\\graphics\\raw_north",
        'difference': f"{north_day}\\graphics\\difference",
        'results': f"{north_day}\\graphics\\results"
    }

    process_loop(north_day, south_day, north_filepath, south_filepath, graphic_outputs)