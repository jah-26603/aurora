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
import yaml
import functions.SpeciesProcessor as SP
import os
import pickle

def process_loop(species, start_file, minus_end_file, north_day, south_day,
                 north_filepath, south_filepath, graphic_outputs, species_info_fp):
    #Initialization
    file_list = glob.glob(f'{north_filepath}/*.nc')
    hemisphere_order = []
    lat_threshold = 50
    dict_list_south_scans = functions.time_window(north_filepath, south_filepath, time_window = (-20, 20))
    keys = list(dict_list_south_scans.keys())
    species_info_dict = yaml.safe_load(open(species_info_fp))
    count = 0
    file_arr = []
    time_arr = []
    species_datastructure = {
        key: {
            "points": np.zeros((len(keys),2)),
            "opoints":np.zeros((len(keys),2)),
            "difference": np.zeros((len(keys), 52, 92)),
            "brightnesses": np.zeros((len(keys), 52, 92)),
            "diff": [], "second_der": [],
            "time_of_scan": []
            }
        for key in list(species_info_dict.keys()) 
        }
    


    for file in tqdm(range(start_file, int(len(keys))- minus_end_file)):
        
        
        try:
            ds = nc.Dataset(file_list[file], 'r')
        except OSError:
            print('Error in reading netcdf4 file')
            continue
        if count == 0:
            count += 1
            day = ds.DATE_START[:10]
            julian_day = file_list[0][21:24]
            print('\n')
            print('Northern Hemisphere Scans from:   ' + day)

        
        #loads data
        latitude = ds.variables['GRID_LAT'][:]    #Note: latitude data is flipped!! So data at top is southern most latitude and data at bottom is Northern most latitude
        radiance = ds.variables['RADIANCE'][:]
        wavelength = ds.variables['WAVELENGTH'][:]
        radiance = np.clip(radiance, 0, np.inf)
        sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
        time = ds.variables['TIME_UTC'][:]
        ema = ds.variables['RAY_SOLAR_PHASE_ANGLE'][:]
        longitude = ds.variables['GRID_LON'][:]
        

        hemisphere_order, hemisphere, *_ = functions.hemisphere(hemisphere_order, sza) #which hemisphere  
        if hemisphere_order[-1] == 1:
            continue
        
        
        for specie in species:
            count+=1
            species_processing = SP(specie, species_info_dict, file, time, radiance, latitude, 
                                    wavelength, dict_list_south_scans, keys)
            processed_info = species_processing.process_species(np.copy(sza), ema, count)
            count+=1
            # print(count)
            # if count> 1:
            #     breakpoint()
            dummy = species_datastructure[specie]
            
            if processed_info is None:
                continue
            
            for key, value in processed_info.items():
                if key in dummy:
                    if isinstance(dummy[key], list):
                        dummy[key].append(value)  # Append to list
                    else:
                        dummy[key][file] = value  # Add to array index
            print(species_processing.time_of_scan)
        if processed_info is not None:
            file_arr.append(file)
            time_arr.append(time)



    
    #remove empty times
    for specie in species:
        for key, value in species_datastructure[specie].items():
            if isinstance(value, np.ndarray):  
                dd = np.copy(value)  
                species_datastructure[specie][key] = dd[file_arr] 

    limb_boundary = functions.track_limb_boundary(species_datastructure,
                                                            species, low_cadence = False)
    
    # dummy_boundary = functions.final_limb_boundary(species_datastructure, estimated_limb_boundary, latitude)
    # limb_boundary = functions.track_limb_boundary(dummy_boundary, species, low_cadence = False)
    
    products_dict = {}
    products_dir = 'products' 
    os.makedirs(products_dir, exist_ok=True)
    
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
        border_image[:,int(limb_boundary[i,0]) :int(limb_boundary[i,1])] = 1
        
        
        dummy_mask_arr = []
        result_arr = []
        auroral_product_results = {'1356': []
                                   , '1493': [],
                                   'LBH':[]}
        
        for specie in species:
            # Create specie-specific subdirectory
            g_out = {
                key: os.path.join(*path.split('\\')[:-1], specie, key) for key, path in graphic_outputs.items()
            }
            
            
            # Ensure directories exist
            for path in g_out.values():
                os.makedirs(path, exist_ok=True)
            
            # Retrieve relevant values from the dictionary
            time_of_scan = species_datastructure[specie]['time_of_scan'][i]
            difference = species_datastructure[specie]['difference'][i]
            brightnesses = species_datastructure[specie]['brightnesses'][i]
            
            # Pass updated g_out to results_loop
            mask, result = functions.results_loop(
                time_of_scan, difference, border_image,
                keys, file_arr[i], latitude, lat_threshold, day,
                brightnesses, limb_boundary[i], g_out, specie
            )
            
            dummy_mask_arr.append(mask)
            result_arr.append(result)
            auroral_product_results[specie] = result
            
    
        dummy_mask_arr = np.array(dummy_mask_arr)
        binary_mask = (np.sum(dummy_mask_arr, axis=0) >= 2).astype(int)
        result_arr = np.array(result_arr)* binary_mask
         
        products = functions.aurora_products(latitude, longitude, time_arr[i], 
                                  auroral_product_results, dummy_mask_arr)
        
        products_dict[i] = products
    with open(f"{products_dir}/products_{julian_day}.pkl", "wb") as f:
        pickle.dump(products_dict, f)
        # plt.figure()
        # offset = 26
        # plt.imshow(result[26:], origin='lower')
        
        # # Extract LBH scatter plot data
        # for slice_info in products["slice_data"]:
        #     if "LBH" in slice_info["species"]:
        #         pwb_lon = slice_info["species"]["LBH"]["pwb_lon"]
        #         ewb_lon = slice_info["species"]["LBH"]["ewb_lon"]
        #         # Scatter plot for PWB and EWB coordinates
        #         plt.scatter(np.where(longitude[52:] == pwb_lon)[1], 
        #                     np.where(longitude[52:] == pwb_lon)[0] - offset + 1, 
        #                     color='white', marker='.', label="PWB" if slice_info["slice_id"] == 0 else "")
        #         plt.scatter(np.where(longitude[52:] == ewb_lon)[1], 
        #                     np.where(longitude[52:] == ewb_lon)[0] - offset - 1, 
        #                     color='deepskyblue', marker='.', label="EWB" if slice_info["slice_id"] == 0 else "")
        
        # # Hide axes and borders
        # plt.axis('off')
        
                
        # # Adjust layout to remove excess whitespace
        # plt.tight_layout()
        
        # plt.show()

        # # breakpoint()
        # plt.figure()
        # plt.imshow(products['O_LBH_ratio'][offset:], origin = 'lower')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.colorbar()
        # plt.show
        # plt.figure()
        # plt.imshow(products['O_N_ratio'][offset:], origin = 'lower')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.colorbar()
        # plt.show()
        # for i in range(len(species)):
        #     g_out = os.path.join(
        #     *str(graphic_outputs['results']).split('\\')[:-1], 
        #     species[i], 'results')
            
        #     functions.save_array(result_arr[i], day, time_of_scan,'results', 
        #                          g_out, species[i], cmap = 'plasma', 
        #                          show_plots = False)

        
        # mask_output = 'D:\\gold_level1c_2020_every_7th\\006\\graphics\\masks'
        # functions.save_array(np.sum(dummy_mask_arr, axis = 0), day, time_of_scan,'mask',
        #                      mask_output, '', cmap = 'plasma',
        #                      show_plots = False, colorbar = True)

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
    
    
    
    
    
    
    
# """plotting stuff"""
# #Zoomed on in auroral region 
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
# im = ax.pcolor(longitude, latitude, result,  # Choose the first dataset for single image
#                 transform=ccrs.PlateCarree(), cmap='plasma')

# # Add a color bar
# cb = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05)
# cb.set_label(f'{units[0]}')  # Use the cor
# plt.show()



import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# # Loop through each slice and create a separate plot
# for slice_info in products["slice_data"]:
#     if "LBH" in slice_info["species"]:
#         fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=-45)})

#         # Set the geographical extent
#         ax.set_extent([-115, 15, 30, 75])

#         # Add gridlines
#         gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                           linewidth=2, color='gray', alpha=0.5, linestyle='--')

#         # Background pcolor plot
#         bb = np.copy(longitude[52:])
#         cc = np.copy(latitude[52:])
#         im = ax.pcolor(bb, cc, result, transform=ccrs.PlateCarree(), cmap='plasma')

#         # Extract scatter plot data
#         pwb = slice_info["species"]["LBH"]["pwb"]
#         ewb = slice_info["species"]["LBH"]["ewb"]
#         pwb_lon = slice_info["species"]["LBH"]["pwb_lon"]
#         ewb_lon = slice_info["species"]["LBH"]["ewb_lon"]

#         # Scatter plot for PWB and EWB coordinates
#         ax.scatter(pwb_lon, pwb, color='red', marker='o', label="PWB", transform=ccrs.PlateCarree(), zorder=3)
#         ax.scatter(ewb_lon, ewb, color='blue', marker='o', label="EWB", transform=ccrs.PlateCarree(), zorder=3)

#         # Add title based on slice_info
#         ax.set_title(f"Slice ID: {slice_info['slice_id']}")

#         # Add a color bar
#         cb = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05)
#         cb.set_label('Rayleighs')

#         # Show or save the figure
#         plt.show()  # Change to `plt.savefig(f"slice_{slice_info['slice_id']}.png")` if you want to save the images


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