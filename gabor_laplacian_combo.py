# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:02:19 2024

@author: dogbl
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Parameters for Gabor filter
ksize = 9  # Size of the Gabor kernel
sigma = 1.0  # Standard deviation of the Gaussian function
theta = np.deg2rad(105)  # Orientation of the Gabor kernel (135 degrees for southwest)
lambd = 5  # Wavelength of the sinusoidal factor
gamma = 0.5  # Aspect ratio of the Gaussian function
psi = 0  # Phase offset

# Create Gabor kernel
gk1 = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
plt.imshow(gk1)

image = dp  # Replace 'path_to_your_image.png' with your image path
filtered_image, kernel = LoG_filter_opencv(image,sigma_x = .9, sigma_y =.9, size_x = 7, size_y = 5)
filtered_image = cv2.convertScaleAbs(filtered_image)
plt.figure()
plt.imshow(filtered_image)
plt.show()
filtered_image = cv2.filter2D(filtered_image, -1, gk1)
plt.figure()
plt.imshow(filtered_image)
plt.show()





