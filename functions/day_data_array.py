# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:07:33 2024

@author: JDawg
"""

import numpy as np

def all_data_to_ang(file, hemisphere_order, sza, brightnesses_LBHS, emission_angle, brightnesses_O,
                    all_LBH, all_O, access, n_all_LBH, n_all_O, access_n, n_indices_LBH,
                    all_Ob, all_brightnesses, all_sza, all_emission_angle, all_longitude, 
                    all_latitude, all_time, longitude, latitude, time_array, filled_indices, dims_ang, lat_vals, long_vals):
    if hemisphere_order[-1] == 1:
        for j in range(sza.shape[0]):
            for k in range(sza.shape[1]):
                if not np.isnan(brightnesses_LBHS[j, k]):
                    if sza[j,k]//1< dims_ang[0] and emission_angle[j,k]//1 < dims_ang[1]:
                        all_LBH[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] += brightnesses_LBHS[j,k]
                        all_O[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] += brightnesses_O[j,k]
                        access[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] +=1
                        
                    else: 
                        pass
        # breakpoint()
        
    else:
        for j in range(sza.shape[0]):
            for k in range(sza.shape[1]):
                if np.isnan(brightnesses_LBHS[j, k]):
                    continue
                else:
                    if sza[j,k]//1< dims_ang[0] and emission_angle[j,k]//1 < dims_ang[1]:
                        n_all_LBH[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] += brightnesses_LBHS[j,k]
                        n_all_O[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] += brightnesses_O[j,k]
                        access_n[file,int(sza[j,k]//1), int(emission_angle[j,k]//1)] +=1
                        n_indices_LBH[file][int(sza[j,k]//1)][ int(emission_angle[j,k]//1)].append([file, j, k])
                        lat_vals[int(sza[j,k]//1)][ int(emission_angle[j,k]//1)].append(latitude[j,k])
                        long_vals[int(sza[j,k]//1)][ int(emission_angle[j,k]//1)].append(longitude[j,k])

                    else: 
                        pass

    '''Storing all of the data of a day into one array'''
    all_Ob[file] = brightnesses_O
    all_brightnesses[file] = brightnesses_LBHS
    all_sza[file] = sza
    all_emission_angle[file] = emission_angle
    all_longitude[file] = longitude
    all_latitude[file] = latitude
    all_time.append( time_array[filled_indices[0,0],filled_indices[0,1]])
    return all_Ob, all_brightnesses, all_sza, all_emission_angle, all_longitude, all_latitude, all_time, all_LBH, all_O, access,n_all_LBH, n_all_O, access_n, n_indices_LBH, lat_vals, long_vals