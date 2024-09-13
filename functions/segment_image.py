# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:18:26 2024

@author: dogbl
"""
import numpy as np
from sklearn.cluster import KMeans


def segment_image (scan, clusters = 3):
    
    data = np.nan_to_num(scan)
    
    
    
    # Step 1: Reshape the data to (num_pixels, 1)
    original_shape = scan.shape
    data = data.reshape(-1, 1)

    # Step 2: Apply KMeans clustering
    n_clusters = clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_

    # Step 3: Reshape the clustered data back to the original image shape
    segmented_image = labels.reshape(original_shape)
    
    return segmented_image, clusters, kmeans