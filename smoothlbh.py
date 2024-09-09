# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:09:45 2024

@author: JDawg
"""
import functions 
import netCDF4 as nc
import numpy as np
import cartopy
import torch
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import cm
import pylab as pl
import statistics
import random
from tabulate import tabulate
import cartopy
from cartopy import config
import cartopy.crs as ccrs
import matplotlib.pylab as plb
from matplotlib import colorbar, colors
from PIL import Image
import glob
from datetime import datetime
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans


file_path = r".\Aurora_Dates\05_11_2024"
# functions.read_tar(file_path)
file_list = glob.glob(f'{file_path}/*.nc')

dims_ang = (91, 91) 
dims_fl = (104,92)


all_O, n_all_O, all_LBH, n_all_LBH, access, access_n = [np.zeros((len(file_list), *dims_ang)) for _ in range(6)]
all_Ob, all_brightnesses, all_sza, all_emission_angle, all_longitude, all_latitude =[np.empty((len(file_list), *dims_fl)) for _ in range(6)]
all_time = []
hemisphere_order = []
n_indices_LBH  = [[[[] for _ in range(91)] for _ in range(91)] for _ in range(len(file_list))]
lat_vals = [[[] for _ in range(91)] for _ in range(91)]
long_vals = [[[] for _ in range(91)] for _ in range(91)]
def fittin(y, a,b,c ,d):
    return (a*y+b*y**.5+ c +d*y**.75)



for file in range(len(file_list)):
# for file in range(1,3):
    ds = nc.Dataset(file_list[file], 'r')
    
    #store data in arrays
    longitude = ds.variables['GRID_LON'][:]
    latitude = ds.variables['GRID_LAT'][:]    #Note: latitude data is flipped!! So data at top is southern most latitude and data at bottom is Northern most latitude
    time = ds.variables['TIME_UTC'][:]
    radiance = ds.variables['RADIANCE'][:]
    radiance_unc = ds.variables['RADIANCE_RANDOM_UNC'][:]  #are these the uncertainties?
    sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
    wavelength = ds.variables['WAVELENGTH'][:]
    emission_angle = ds.variables['EMISSION_ANGLE'][:]
    
    
    radiance = np.clip(radiance, 0, np.inf)
    hemisphere_order, hemisphere, skip_s, skip_n = functions.hemisphere(hemisphere_order, sza, skip_s = False, skip_n = False) #which hemisphere
    
    if skip_s == 1 and hemisphere_order[-1] == 1:
        continue
    if skip_n == 1 and hemisphere_order[-1] == 0:
        continue
    
    
    filled_indices, one_pixel = functions.filled_indices(wavelength) #acceptable indices for analysis
    date, time_array =  functions.date_and_time(filled_indices, time) #gets date and time
    
    brightnesses_LBHS, brightnesses_LBHS_unc = functions.get_data_info(radiance, radiance_unc, one_pixel, 138, 152, 148, 150, multi_regions= True)    
    brightnesses_O, brightnesses_O_unc = functions.get_data_info(radiance, radiance_unc, one_pixel, 135, 136, np.nan, np.nan)


    all_Ob, all_brightnesses, all_sza, all_emission_angle, all_longitude, all_latitude, all_time, all_LBH, all_O, access, n_all_LBH, n_all_O, access_n, n_indices_LBH, lat_vals, long_vals = functions.all_data_to_ang(
            file, hemisphere_order, sza, brightnesses_LBHS, emission_angle, brightnesses_O,
            all_LBH, all_O, access, n_all_LBH, n_all_O, access_n, n_indices_LBH, all_Ob,
            all_brightnesses, all_sza, all_emission_angle, all_longitude, all_latitude, 
            all_time, longitude, latitude, time_array, filled_indices, dims_ang, lat_vals, long_vals
                        )
    
    
    functions.plot_on_globe(latitude, longitude, brightnesses_LBHS, date, time_array, filled_indices, 'LBHS', hemisphere_order, skip_south_plot = True)
    # '''plotting'''
    # set_max_value(brightnesses_LBHS, 6000)
    # levels = [0,150, 300, 450,600, 1200, 1800, 2400, 3200, 4000, 4800,5000]

    # plt.figure()
    # plt.contourf(sza, emission_angle, brightnesses_LBHS, levels = levels)
    # plt.xlim(round(np.nanmin(sza)), round(np.nanmax(sza)))
    # plt.ylim(round(np.nanmin(emission_angle)), round(np.nanmax(emission_angle)))
    # plt.title(f'{hemisphere} LBH vs SZA and Emission Angle ' +
    #           'Date: '+date+'\nTime: '+ time_array[filled_indices[0,0],filled_indices[0,1]]
    #               )
    # plt.xlabel('SZA (degrees)')
    # plt.ylabel('Emission Angle (degrees)')
    # plt.xlim(0,120)
    # plt.ylim(0,90)
    # plt.colorbar()
    # plt.show()
    
    # plt.figure()
    # plt.contourf(sza, emission_angle, brightnesses_O)
    # plt.xlim(round(np.nanmin(sza)), round(np.nanmax(sza)))
    # plt.ylim(round(np.nanmin(emission_angle)), round(np.nanmax(emission_angle)))
    # plt.title(f'{hemisphere} LBH vs SZA and Emission Angle ' +
    #           'Date: '+date+'\nTime: '+ time_array[filled_indices[0,0],filled_indices[0,1]]
    #               )
    # plt.xlabel('SZA (degrees)')
    # plt.ylabel('Emission Angle (degrees)')
    # plt.xlim(0,120)
    # plt.ylim(0,90)
    # plt.colorbar()
    # plt.show()

    

smooth_lbh, smooth_O, n_smooth, n_smooth_O, smooth_dif, smooth_dif_O = [np.zeros(dims_ang) for _ in range(6)]


#All frames in the day summed northern hemisphere
n_smooth = np.nan_to_num(np.sum(n_all_LBH, axis = 0) / np.sum(access_n, axis = 0), nan = 0)
n_smooth_O = np.nan_to_num(np.sum(n_all_O, axis = 0) / np.sum(access_n, axis = 0), nan = 0)


#All frames in the day summed southern hemisphere
smooth_lbh = np.clip(np.nan_to_num(np.sum(all_LBH, axis = 0)/ np.sum(access, axis = 0) , nan = 0), a_min = 0, a_max = 6000)
smooth_O = np.clip(np.nan_to_num(np.sum(all_O, axis = 0)/ np.sum(access, axis = 0) , nan = 0 ), a_min = 0, a_max = 14000)


x_1 = np.linspace(0,90,91) #summer
x_2 = np.linspace(0, 90, 91)
data = smooth_lbh.T
smooth_lbh_fit = [0]*91
data1 = smooth_O.T
smooth_O_fit = [0]*91


for i in range(91): 
    useful = data[i]
    domain_useful = x_1
    mask = useful!= 0
    useful = useful[mask]
    domain_useful= domain_useful[mask]
    popt, pcov = curve_fit(fittin, domain_useful, useful)
    fit = fittin(x_1, *popt)
    smooth_lbh_fit[i] = fit

        
for i in range(91): 
    useful = data1[i]
    domain_useful = x_1
    mask = useful!= 0
    useful = useful[mask]
    domain_useful= domain_useful[mask]
    popt, pcov = curve_fit(fittin, domain_useful, useful)
    fit = fittin(x_1, *popt)
    smooth_O_fit[i] = fit





smooth_O_fit = np.clip(smooth_O_fit, 0, 14000)
smooth_lbh_fit = np.clip(smooth_lbh_fit, 0 , 6000)


q = np.nanmedian(n_smooth[:50,:])/np.nanmedian(smooth_lbh_fit[:50,:])
q1 = np.nanmedian(n_smooth_O[:50,:])/np.nanmedian(smooth_O_fit[:50,:])      
  
for j in range(91):
    for k in range(91):
        b = n_smooth[j,k] - q*smooth_lbh_fit[k,j]
        if np.isnan(b) or b<0:
            smooth_dif[j,k] = 0
        else:
            smooth_dif[j,k] = b

            
            
for j in range(91):
    for k in range(91):
        c = n_smooth_O[j,k] - q1*smooth_O_fit[k,j]
        if np.isnan(c) or c<0:
            smooth_dif_O[j,k] =0 
        else: 
            smooth_dif_O[j,k] = c



     
new_lev = [0,50,100,150,200,250,300]
levels = [0, 100, 200, 300, 400, 500,600, 700, 800, 900, 1000, 2000, 3000,4000,5000]

smooth_dif1 = np.clip(smooth_dif, 0, 600)

plt.figure()
plt.contourf(np.linspace(0, 90, 91), np.linspace(0, 90, 91), smooth_dif1.T, levels = new_lev)
plt.colorbar().set_label('R')
plt.ylim(0, np.nanmax(emission_angle))
plt.xlim(0,90)
plt.title(f'Difference North-South LBH vs. SZA and EMA {date}')
plt.xlabel('SZA')
plt.ylabel('EMA')
plt.show()

            
plt.figure()
plt.contourf(x_1, x_2, smooth_lbh_fit, levels = levels)
plt.colorbar().set_label('R')
plt.ylim(0, np.nanmax(emission_angle))
plt.xlim(0,90)
plt.title(f'Smooth Fit South LBH vs. SZA and EMA {date}')
plt.xlabel('SZA')
plt.ylabel('EMA')
plt.show()






plt.figure()
c1 = plt.contourf(np.linspace(0, 90, 91), np.linspace(0, 90, 91), n_smooth.T,levels = levels).levels
plt.colorbar().set_label('R')
plt.ylim(0, np.nanmax(emission_angle))
plt.xlim(0,90)
plt.title(f'North LBH vs. SZA and EMA {date}')
plt.xlabel('SZA')
plt.ylabel('EMA')
plt.show()


plt.figure()
c2 = plt.contourf(np.linspace(0, 90, 91), np.linspace(0, 90, 91), smooth_lbh.T, levels = levels).levels
plt.colorbar().set_label('R')
plt.ylim(0, np.nanmax(emission_angle))
plt.xlim(0,90)
plt.title(f'Southern LBH vs. SZA and EMA {date}')
plt.xlabel('SZA')
plt.ylabel('EMA')
plt.show()


from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve, median_filter

smoothed_data = gaussian_filter(smooth_dif1, sigma=5)

# plt.imshow(smoothed_data)
pp = smooth_dif1
pp[:40] = 0
pp[:,:75] = 0
qq = peak_local_max(pp, min_distance = 0, threshold_abs = 100)
# qq = peak_local_max(smooth_dif1[:40], min_distance = 0, threshold_abs = 20)
plt.figure()
plt.imshow(smooth_dif1)
plt.scatter(qq[:,1], qq[:,0])
plt.show()

import itertools
lat_list = [[val for val in lat_vals[qq[i,1]][qq[i,0]]] for i in range(qq.shape[0])]
lat_list = list(itertools.chain.from_iterable(lat_list))
lat_list = np.array(lat_list)

long_list = [[val for val in long_vals[qq[i,1]][qq[i,0]]] for i in range(qq.shape[0])]
long_list = list(itertools.chain.from_iterable(long_list))
long_list = np.array(long_list)


pos_array = np.column_stack((lat_list, long_list))
df = pd.DataFrame(pos_array)
df = df.dropna()
pos_array = df.to_numpy()

plt.figure()
# plt.imshow(brightnesses_O)
plt.scatter(pos_array[:,1], pos_array[:,0])
plt.show()

# #each scan? minus the entire day... kinda sucks
# for i in range(12):
#     mask = n_all_LBH[i] != 0
#     sub_array = np.where(mask, smooth_lbh_fit[i], 0)
    
    
#     plt.figure()
#     plt.imshow(n_all_LBH[i] - sub_array)
#     plt.show()



