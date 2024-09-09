# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:37:58 2024

@author: JDawg
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

def plot_on_globe(latitude, longitude, brightnesses, date, time_array, filled_indices, species, hemisphere_order, skip_south_plot  = False, vmax = 6000 ):
    if hemisphere_order[-1] and skip_south_plot == 1:
        pass
    else:
        
        plt.figure()
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
            "savefig.edgecolor": "white"})
        plt.rcParams['axes.labelpad']=15
    
        #'''Setup the size of the figure'''
        fig= plt.figure(figsize=(7,6))
    
        #'''Title'''
        fig.suptitle(f'{species} Radiance Colormaps \nDate: '+date+'\nTime: '+ time_array[filled_indices[0,0],filled_indices[0,1]],color='black', y=1, weight='bold')
    
        #'''Setup the projection of your plot, I use 'geostationary' as a example, you can also try NearsidePerspective if you want to focus on other latitue '''
        Nearside_ax=ccrs.NearsidePerspective(central_latitude=0,central_longitude=-47.5,satellite_height=35786000)
    
        #'''Simply using pcolor with a proper projection, you can make the plot with data on the globe.'transform=ccrs.PlateCarree()' place the data on accordingly in a geographic longitude vs. latitude coordinate'''
        ax=plt.subplot(projection=Nearside_ax)  #geostationary_ax)
        im=ax.pcolor(longitude, latitude, brightnesses, transform=ccrs.PlateCarree(),cmap='plasma', vmin=0, vmax = vmax)
    
    
        #'''Creat a color bar'''
        cb=plt.colorbar(im, pad = 0.1)
        cb.set_label(f'{species} Radiance (Rayleighs)')
    
        ax.set_global()
        gl = ax.gridlines(color='lightgrey', draw_labels=True)
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}
    
        plt.legend()
        plt.show()
        
        
        fig= plt.figure(figsize=(7,4))
        #fig.suptitle('Date: '+date+'\nTime: '+ time_array[filled_indices[0,0],filled_indices[0,1]],color='black', y=1)
    
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=longitude[45, 51]))
        ax.set_extent([-115, 15, 30, 80])
    
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
        im=ax.pcolor(longitude, latitude, brightnesses, transform=ccrs.PlateCarree(),cmap='plasma', vmin=0, vmax=vmax) 
    
    
        #'''Creat a color bar'''
        cb=plt.colorbar(im, pad = 0.1)
        cb.set_label(f'{species} Radiance (Rayleighs)')
    
        #ax.set_global()
        #gl = ax.gridlines(color='lightgrey', draw_labels=True)
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
    
        plt.legend()
        plt.show()