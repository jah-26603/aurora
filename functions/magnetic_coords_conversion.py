# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:45:12 2025

@author: JDawg
"""

import numpy as np

def geographic_to_magnetic_lat(lat, lon, mag_lat_pole = 80.65, mag_lon_pole = -72.62):
    """
    Convert geographic latitude to magnetic latitude.

    Parameters:
        lat (float): Geographic latitude (degrees).
        lon (float): Geographic longitude (degrees).
        mag_lat_pole (float): Magnetic pole latitude (degrees).
        mag_lon_pole (float): Magnetic pole longitude (degrees).

    Returns:
        float: Magnetic latitude (degrees).
    """
    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    mag_lat_pole_rad = np.radians(mag_lat_pole)
    mag_lon_pole_rad = np.radians(mag_lon_pole)

    # Compute magnetic latitude
    delta_lon = lon_rad - mag_lon_pole_rad
    sin_lat_m = (np.cos(delta_lon) * np.cos(lat_rad) * np.cos(mag_lat_pole_rad)+ np.sin(lat_rad) * np.sin(mag_lat_pole_rad)
    )
    lat_m_rad = np.arcsin(sin_lat_m)

    # Convert back to degrees
    return np.degrees(lat_m_rad)



def compute_mlt(longitude, ut_hours, mag_pole_lon = -72.62):
    
    # Calculate magnetic longitude
    magnetic_longitude = longitude - mag_pole_lon
    magnetic_longitude = (magnetic_longitude + 180) % 360 - 180
    
    # Calculate MLT
    mlt = (magnetic_longitude / 15) + ut_hours
    mlt = mlt % 24
    return mlt, magnetic_longitude

# Example usage
geographic_lat = 45.0  # in degrees
geographic_lon = -75.0  # in degrees
magnetic_pole_lat = 80.65  # 2020 North magnetic pole latitude
magnetic_pole_lon = -72.62  # 2020 North magnetic pole longitude
ut_hours = 15

magnetic_lat = geographic_to_magnetic_lat(
    geographic_lat, geographic_lon, magnetic_pole_lat, magnetic_pole_lon
)
mlt = compute_mlt(geographic_lon, ut_hours, magnetic_pole_lon)

