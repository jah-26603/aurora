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


file_path = r".\Aurora_Dates\09_21_2022"
functions.read_tar(file_path)
file_list = glob.glob(f'{file_path}/*.nc')

dims_ang = (91, 91)
dims_fl = (104,92)


all_O, n_all_O, all_LBH, n_all_LBH, access, access_n = [np.zeros((len(file_list), *dims_ang)) for _ in range(6)]
all_Ob, all_brightnesses, all_sza, all_emission_angle, all_longitude, all_latitude =[np.empty((len(file_list), *dims_fl)) for _ in range(6)]
all_time = []
hemisphere_order = []
n_indices_LBH = keep_track_of_indices = [[[[] for _ in range(91)] for _ in range(91)] for _ in range(len(file_list))]

breakpoint()

def fittin(y, a,b,c ,d):
    return (a*y+b*y**.5+ c +d*y**.75)

def set_max_value(arr, max_val):
    arr[arr > max_val] = max_val


for file in range(len(file_list)):
# for file in range(1,3):
    name = file_list[file]
    ds = nc.Dataset(name, 'r')
    
    
    
    #store data in arrays
    longitude = ds.variables['GRID_LON'][:]
    latitude = ds.variables['GRID_LAT'][:]    #Note: latitude data is flipped!! So data at top is southern most latitude and data at bottom is Northern most latitude
    time = ds.variables['TIME_UTC'][:]
    radiance = ds.variables['RADIANCE'][:]
    radiance_unc = ds.variables['RADIANCE_RANDOM_UNC'][:]  #are these the uncertainties?
    sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
    wavelength = ds.variables['WAVELENGTH'][:]
    emission_angle = ds.variables['EMISSION_ANGLE'][:]
    
    first_row_unpopulated = np.isnan(sza[0].data).all()   #Boolean: if True, first row of sza matrix is not populated
    last_row_unpopulated = np.isnan(sza[len(sza) - 1].data).all()   #Boolean: if True, last row of sza matrix is not populated
    
    #We want values in bottom half of data - corresponds to northern hemisphere of Earth (latitude data is flipped)
    if last_row_unpopulated and not first_row_unpopulated:
        print('This is a Southern Hemisphere Scan')
        hemisphere = 'Southern'
        hemisphere_order.append(1)
        # continue
        
    if first_row_unpopulated and not last_row_unpopulated :
        print('This is a Northern Hemisphere Scan')
        hemisphere = 'Northern'
        hemisphere_order.append(0)
        # continue
    
    #Find i,j indices of wavelengths that are filled with data
    filled_indices = []
    for i in range(wavelength.shape[0]):
        for j in range(wavelength.shape[1]):
            if np.isfinite(wavelength[i][j]).any():
                filled_indices.append((i, j))
    filled_indices = np.array(filled_indices)
    
    radiance = np.clip(radiance,0,np.inf)   #For all negative radiances, round up to 0
    one_pixel = np.array(wavelength[filled_indices[0][0],filled_indices[0][1],:]) #array of 800 wavelength values for one pixel
    
    #Extract date and times
    date = (''.join([c.decode('utf-8') for c in time[filled_indices[0,0],filled_indices[0,1]]]))[:10]
    time_array = np.array([[None]*92]*104)
    for index in filled_indices:
        i, j = index
        time_array[i,j] = (''.join([c.decode('utf-8') for c in time[i, j]]))[11:]
        
        
        
    #Short
    #Find wavelength k indices that are within 1380-1530, except 1480-1500 (emission for N2 LBH-Short bands)
    indices_of_interest_LBHS = np.where(np.logical_and(one_pixel >= 138, one_pixel <= 152))[0]
    indices_of_interest_LBHS = indices_of_interest_LBHS[~np.logical_and(one_pixel[indices_of_interest_LBHS] > 148, one_pixel[indices_of_interest_LBHS] < 150)]
    
    
    
    #For each pixel, find radiance values at ^these indices, sum them, multiply by 0.04 to get brightness values
    radiances_of_interest_LBHS = np.asarray([[[np.NaN]*len(indices_of_interest_LBHS)]*92]*104)
    radiances_of_interest_LBHS_unc = np.asarray([[[np.NaN]*len(indices_of_interest_LBHS)]*92]*104)
    brightnesses_LBHS = np.asarray([[np.NaN]*92]*104)
    brightnesses_LBHS_unc = np.asarray([[np.NaN]*92]*104)
    
    for i in range(104):
        for j in range(92):
            for k1, k2 in enumerate(indices_of_interest_LBHS):
                radiances_of_interest_LBHS[i, j, k1] = radiance[i,j,k2]
                radiances_of_interest_LBHS_unc[i, j, k1] = radiance_unc[i,j,k2]
    
            ###Take greatest 50% of these radiances, turn rest into zeros, keep x and y locations of everything    
            mean_radiance = np.nanmean(radiances_of_interest_LBHS[i, j])
            mean_radiance_unc = np.nanmean(radiances_of_interest_LBHS_unc[i, j])
    
            for k, rad in enumerate(radiances_of_interest_LBHS):
                if radiances_of_interest_LBHS[i,j,k] < mean_radiance:
                    radiances_of_interest_LBHS[i,j,k] = 0.0
                    radiances_of_interest_LBHS_unc[i,j,k] = 0.0
    
            brightnesses_LBHS[i,j] = 0.04*radiances_of_interest_LBHS[i, j, :].sum()  #0.04 is R/nm conversion
            brightnesses_LBHS_unc[i,j] = 0.04*math.sqrt((radiances_of_interest_LBHS_unc[i, j, :]**2).sum())

    #-------Atomic Oxygen-------
    #Find wavelength k indices that are within 1350-1360 (emission for atomic Oxygen: 135.6 nm)
    indices_of_interest_O = np.where(np.logical_and(one_pixel >= 135, one_pixel <= 136))[0]  

    #For each pixel, find radiance values at ^these indices, sum them, multiply by 0.04 to get brightness values
    radiances_of_interest_O = np.asarray([[[np.NaN]*len(indices_of_interest_O)]*92]*104)
    brightnesses_O = np.asarray([[np.NaN]*92]*104)
    for i in range(104):
        for j in range(92):
            for k1, k2 in enumerate(indices_of_interest_O):
                radiances_of_interest_O[i, j, k1] = radiance[i,j,k2]
            brightnesses_O[i,j] = 0.04*radiances_of_interest_O[i, j, :].sum()  #0.04 is R/nm conversion
                
                
                
    qqqq = brightnesses_LBHS**.5            
    plt.figure()
    plt.imshow(np.flip(qqqq, axis =0))
    plt.xlim(0,89)
    plt.ylim(0,55)
    plt.show()

    '''plotting'''
    # set_max_value(brightnesses_LBHS, 5000)
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
    
    
    '''Storing all of the data of a day into one array'''
    all_Ob[file] = brightnesses_O
    all_brightnesses[file] = brightnesses_LBHS
    all_sza[file] = sza
    all_emission_angle[file] = emission_angle
    all_longitude[file] = longitude
    all_latitude[file] = latitude
    all_time.append( time_array[filled_indices[0,0],filled_indices[0,1]])
    
    
    if hemisphere_order[file] == 1:
        for j in range(sza.shape[0]):
            for k in range(sza.shape[1]):
                if np.isnan(brightnesses_LBHS[j, k]):
                    pass
                else:
                    if sza[j,k]//1< 91 and emission_angle[j,k]//1 < 91:
                        all_LBH[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] += brightnesses_LBHS[j,k]
                        all_O[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] += brightnesses_O[j,k]
                        access[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] +=1
                        
                    else: 
                        pass
                    
                
    else:
        for j in range(sza.shape[0]):
            for k in range(sza.shape[1]):
                if np.isnan(brightnesses_LBHS[j, k]):
                    continue
                else:
                    if sza[j,k]//1< 91 and emission_angle[j,k]//1 < 91:
                        n_all_LBH[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] += brightnesses_LBHS[j,k]
                        n_all_O[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] += brightnesses_O[j,k]
                        access_n[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] +=1
                        n_indices_LBH[file][int(sza[j,k]//1)][ int(emission_angle[j,k]//1)].append([i,j,k])
                    else: 
                        pass


smooth_lbh =np.zeros((91,91))
smooth_n_lbh = np.zeros((len(file_list),91,91))
smooth_O =np.zeros((91,91))
smooth_n_O = np.zeros((len(file_list),91,91))

n_smooth= np.zeros((91,91))
n_smooth_O = np.zeros((91,91))



nn_access = np.zeros((91,91))
nn_LBH = np.zeros((91,91))
nn_O = np.zeros((91,91))

smooth_dif = np.zeros((91,91))
smooth_dif_O = np.zeros((91,91))

for i in range(access.shape[0]):
    for j in range(91):
        for k in range(91):
            if np.isnan(all_LBH[i,j,k]):
                continue
            else:
                nn_access[j,k] += access[i,j,k]
                nn_LBH[j,k] += all_LBH[i,j,k]
                nn_O[j,k] += all_O[i,j,k]

for i in range(access.shape[0]):
    for j in range(91):
        for k in range(91):
            if access_n[i,j,k] ==0:
                continue
            else:
                smooth_n_lbh[i,j,k] = n_all_LBH[i,j,k]/access_n[i,j,k]
                smooth_n_O[i,j,k] = n_all_LBH[i,j,k]/access_n[i,j,k]

for j in range(91):
    for k in range(91):
        b = np.sum(access_n[:,j,k])            
        m = np.sum(n_all_LBH[:,j,k])
        dum1 = np.sum(n_all_O[:,j,k])
        if b == 0:
            continue
        else:
            n_smooth[j,k] = m/b
            n_smooth_O[j,k] = dum1/b
        
for j in range(91):
    for k in range(91):
        if nn_access[j,k] == 0:
            continue
        else:
            x = nn_LBH[j,k] / nn_access[j,k]
            if x > 6000:
                smooth_lbh[j,k] = 6000
            else:
                smooth_lbh[j,k] = x
            y = nn_O[j,k] / nn_access[j,k]
            if y>14000:
                smooth_O[j,k] = 14000
            else:
                smooth_O[j,k] = y
                
                

levels = [0, 100, 200, 300, 400, 500,600, 700, 800, 900, 1000, 2000, 3000,4000,5000]

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



# plt.figure()
# plt.contourf(np.linspace(0, 90, 91), np.linspace(0, 90, 91), n_smooth_O.T)
# plt.colorbar().set_label('R')
# plt.ylim(0, np.nanmax(emission_angle))
# plt.xlim(0,90)
# plt.title(f'Raw North LBH vs. SZA and EMA {date}')
# plt.xlabel('SZA')
# plt.ylabel('EMA')
# plt.show()


# plt.figure()
# plt.contourf(np.linspace(0, 90, 91), np.linspace(0, 90, 91), smooth_O.T)
# plt.colorbar().set_label('R')
# plt.ylim(0, np.nanmax(emission_angle))
# plt.xlim(0,90)
# plt.title(f'Southern LBH vs. SZA and EMA {date}')
# plt.xlabel('SZA')
# plt.ylabel('EMA')
# plt.show()


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
    a_opt,b_opt, c_opt, d_opt = popt
    fit = fittin(x_1, a_opt,b_opt, c_opt, d_opt)
    # fit[~mask] = 0
    smooth_lbh_fit[i] = fit

        
for i in range(91): 
    useful = data1[i]
    domain_useful = x_1
    mask = useful!= 0
    useful = useful[mask]
    domain_useful= domain_useful[mask]
    popt, pcov = curve_fit(fittin, domain_useful, useful)
    a_opt,b_opt, c_opt, d_opt = popt
    fit = fittin(x_1, a_opt,b_opt, c_opt, d_opt)
    # fit[~mask] = 0
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
# for i in range(20):
#     for k in range(91):
#         keep_me_for_now[k,i] = 0
        
# q = np.nanmean(n_smooth)/np.nanmean(keep_me_for_now  ) 
# for j in range(91):
#     for k in range(91):
#         b = n_smooth[j,k] - q*keep_me_for_now[k,j]
#         if np.isnan(b) or b<0:
#             smooth_dif[j,k] = 0
#         else:
#             smooth_dif[j,k] = b    


     
new_lev = [0,50,100,150,200,250,300]

smooth_dif1 = np.clip(smooth_dif, 0, 300)

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





            



# plt.figure()
# plt.contourf(np.linspace(0, 90, 91), np.linspace(0, 90, 91), smooth_dif_O.T)
# plt.colorbar().set_label('R')
# plt.ylim(0, np.nanmax(emission_angle))
# plt.xlim(0,90)
# plt.title(f'Difference North-South O vs. SZA and EMA {date}')
# plt.xlabel('SZA')
# plt.ylabel('EMA')
# plt.show()

            


# smooth_dif_O1 = np.clip(smooth_dif_O, 0, 500)


# plt.figure()
# plt.contourf(np.linspace(0, 90, 91), np.linspace(0, 90, 91), smooth_dif_O1.T)
# plt.colorbar().set_label('R')
# plt.ylim(0, np.nanmax(emission_angle))
# plt.xlim(0,90)
# plt.title(f'Difference North-South O vs. SZA and EMA {date}')
# plt.xlabel('SZA')
# plt.ylabel('EMA')
# plt.show()

            
# plt.figure()
# plt.contourf(x_1, x_2, smooth_O_fit)
# plt.colorbar().set_label('R')
# plt.ylim(0, np.nanmax(emission_angle))
# plt.xlim(0,90)
# plt.title(f'Smooth Fit Southern O vs. SZA and EMA {date}')
# plt.xlabel('SZA')
# plt.ylabel('EMA')
# plt.show()

# plt.figure()
# plt.contourf(x_1, x_2, n_smooth_O.T)
# plt.colorbar().set_label('R')
# plt.ylim(0, np.nanmax(emission_angle))
# plt.xlim(0,95)
# plt.title(f'Northern Hemisphere O vs. SZA and EMA {date}')
# plt.xlabel('SZA')
# plt.ylabel('EMA')
# plt.show()




# count = 0

# # number of southern hemisphere scans
# for scan in hemisphere_order: 
#     if scan == 1:
#         count += 1

# new_ema_axis = np.linspace(60,90,91)
# new_sza_axis = np.linspace(0,120,361)        
# new_all_data = [[[[] for _ in range(len(new_sza_axis))] for _ in range(len(new_ema_axis))] for _ in range(len(file_list))]
# keep_track_of_indices = [[[[] for _ in range(len(new_sza_axis))] for _ in range(len(new_ema_axis))] for _ in range(len(file_list))]

# q1 = (max(new_ema_axis) - min(new_ema_axis))/(len(new_ema_axis)-1)
# q2 = (max(new_sza_axis) - min(new_sza_axis))/ (len(new_sza_axis)-1)
# # convert every single array into a 200x800 array        
# for i in range(len(file_list)): 
#     for j in range(104):
#         for k in range(92):
#             if np.isnan(all_brightnesses[i,j,k]) or all_emission_angle[i,j,k] < 60 or all_sza[i,j,k] > 120:
#                 continue
#             else:
#                 # brightness of each slice
#                 new_all_data[i][int((all_emission_angle[i,j,k]-60)//q1)][int(all_sza[i,j,k]//q2)].append(all_brightnesses[i,j,k]) 
#                 #location of each input in the original data array
#                 keep_track_of_indices[i][int((all_emission_angle[i,j,k]-60)//q1)][int(all_sza[i,j,k]//q2)].append([i,j,k])
    
# # now to convert all of these entries into single elements    
# New_smooth_all_data = np.zeros((len(file_list),len(new_ema_axis),len(new_sza_axis)))
# for i in range(len(file_list)):
#     for j in range(len(new_ema_axis)):
#         for k in range(len(new_sza_axis)):
#             New_smooth_all_data[i,j,k] = np.nanmean(new_all_data[i][j][k][:])
    
    
# '''How to get the average southern hemisphere fit?'''
# count = 0
# for i in range(len(hemisphere_order)):
#     if hemisphere_order[i] ==1:
#         count+=1
        
# southern_hemisphere = np.zeros((count, len(new_ema_axis), len(new_sza_axis)))
# north_hemisphere = np.zeros((len(file_list) - count, len(new_ema_axis), len(new_sza_axis)))



# if hemisphere_order[0] ==1 and hemisphere_order[-1] == 1:
#     for i in range(count-1):
#         southern_hemisphere[i] = New_smooth_all_data[int(2*i)]
#     for i in range(len(file_list)-count-2):
#         north_hemisphere[i] = New_smooth_all_data[int(2*i+1)]
        
# elif hemisphere_order[0] ==0 and hemisphere_order[-1] == 0:        
#     for i in range(count-1):
#         north_hemisphere[i] = New_smooth_all_data[int(2*i)]
#     for i in range(len(file_list)-count-2):
#         southern_hemisphere[i] = New_smooth_all_data[int(2*i+1)]

        
        
# southern_hemisphere[np.isnan(southern_hemisphere)] = 0         

# south_avg =np.empty_like(southern_hemisphere)
# for i in range(len(southern_hemisphere) - 1):
#     for j in range(len(new_ema_axis)):
#         for k in range(len(new_sza_axis)):
#             avg = .5*(southern_hemisphere[i,j,k]+ southern_hemisphere[i+1,j,k])
#             south_avg[i,j,k] = avg
    
# smooth_south_avg = np.empty_like(south_avg)

# "now to curve fit every southern image"

# for i in range(len(southern_hemisphere)-1):
#     for j in range(len(new_ema_axis)):
#         useful = south_avg[i,j,:]
#         domain_useful = new_sza_axis
#         mask = useful!= 0
#         useful = useful[mask]
#         domain_useful= domain_useful[mask]
#         if len(useful)> 4:
#             popt, pcov = curve_fit(fittin, domain_useful, useful)
#             a_opt,b_opt, c_opt, d_opt = popt
#             fit = fittin(new_sza_axis, a_opt,b_opt, c_opt, d_opt)
#             fit[~mask] = 0
#             smooth_south_avg[i,j] = fit
    
# smooth_south_avg = np.maximum(smooth_south_avg,0)




