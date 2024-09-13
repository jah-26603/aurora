# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:37:58 2024

@author: JDawg
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np
from matplotlib.path import Path

def plot_on_globe(latitude, longitude, brightnesses, date, time_array, filled_indices, species, hemisphere_order, skip_south_plot  = False, vmax = [1000, 1000, 3, 3], units = ['Rayleighs (R)', 'Rayleighs (R)', 'Classes', 'Classes']):
    if hemisphere_order[-1] and skip_south_plot == 1:
        pass
    else:
        

        # Update the default Matplotlib parameters
        plt.rcParams.update({
            "font.size": 10,
            "lines.color": "black",
            "patch.edgecolor": "black",
            "text.color": "black",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "lightgray",
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white"
        })
        plt.rcParams['axes.labelpad'] = 15
        
        # Setup the figure size and title
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': ccrs.NearsidePerspective(central_latitude=0, central_longitude=-47.5, satellite_height=35786000)})
        fig.suptitle(f'Radiance Colormaps \nDate: {date}\nTime: {time_array[filled_indices[0,0], filled_indices[0,1]]}', color='black', y=1.05, weight='bold')
        plt.subplots_adjust(wspace=0.4, hspace= 1)
        # Loop through the 2x2 axes grid
        for i, ax in enumerate(axes.flat):
            im = ax.pcolor(longitude, latitude, brightnesses[i], transform=ccrs.PlateCarree(), cmap='plasma', vmin=0, vmax=vmax[i])
            ax.set_title(species[i], weight = 'bold', fontsize = 12)
            # Create color bar for each subplot
            cb = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.1)
            cb.set_label(f'{units[i]}')
        

            # Customize gridlines and global view
            gl = ax.gridlines(color='lightgrey', draw_labels=True)
            gl.xlabel_style = {'size': 10, 'color': 'black'}
            gl.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}

        plt.tight_layout()
        plt.show()
        
        

        # Setup figure with multiple subplots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=longitude[0, 51])})
        fig.suptitle(f'Radiance Colormaps Zoomed \nDate: {date}\nTime: {time_array[filled_indices[0,0], filled_indices[0,1]]}', color='black', y=1.05, weight='bold')
        plt.subplots_adjust(wspace=0.4, hspace= 1)

        # Loop through the 2x2 axes grid
        for i, ax in enumerate(axes.flat):
            ax.set_extent([-115, 15, 30, 80])
            ax.set_title(species[i], weight = 'bold', fontsize = 12)
            # Create gridlines for each subplot
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=2, color='gray', alpha=0.5, linestyle='--')
        
            # Plot the data for each subplot
            im = ax.pcolor(longitude, latitude, brightnesses[i], transform=ccrs.PlateCarree(), cmap='plasma', vmin=0, vmax=vmax[i])
        
            # Create color bar for each subplot
            cb = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.1)
            cb.set_label(f'{units[i]} ')
        
            # Customize the gridline label styles
            gl.xlabel_style = {'size': 10, 'color': 'black'}
            gl.ylabel_style = {'size': 10, 'color': 'black'}
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()