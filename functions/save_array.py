# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:40:37 2024

@author: JDawg
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def save_array(image, day, time, name, save_filepath, species='', 
               cmap='viridis', show_plots=True, colorbar = False):
    
    # Flip and rotate the image
    image = np.fliplr(np.rot90(image, k=2))
    time = time[:5].replace(":", "_")

    # Ensure the directory exists
    if not os.path.exists(save_filepath):
        os.makedirs(save_filepath)
    
    # Create figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image, cmap=cmap)
    ax.axis('off')  # Turn off the axes

    # Add text annotations
    text_str = fr"{day} {time}"
    ax.text(0.5, 0.05, text_str, fontsize=12, color='white', 
            bbox=dict(facecolor='black', alpha=0.5), ha='center', va='bottom', 
            transform=ax.transAxes)
    if colorbar:
        # Create legend patches
        legend_patches = [
            mpatches.Patch(color='yellow', label='3'),
            mpatches.Patch(color='orange', label='2'),
            mpatches.Patch(color='purple', label='1')
        ]
    
        # Add legend box in bottom right
        ax.legend(handles=legend_patches, loc='lower right', fontsize=10, 
                  framealpha=0.8, edgecolor='white')
    # Save the figure
    plt.savefig(rf'{save_filepath}\{name}{time}.png', bbox_inches='tight', pad_inches=0)

    # Show or close the plot
    if show_plots:
        plt.show()
        
    else:
        plt.close()


    


def save_plots(plots, plot_labels, day, time, name, save_filepath, show_plots = True):
    time = time[:5]
    time = time.replace(":", "_")
    #Check if dir exists
    if not os.path.exists(save_filepath):
        os.makedirs(save_filepath)
        
    # Create a figure and display the array
    plt.figure()
    for i in range(len(plots)):
        plt.plot(plots[i] + i, label = rf'{plot_labels[i]}')
    
    # Add a text box at the center bottom
    text_str = fr"{day} {time}"
    plt.text(0.5, 0.05, text_str, fontsize=12, color='white', 
             bbox=dict(facecolor='black', alpha=0.5), ha='center', va='bottom', 
             transform=plt.gca().transAxes)
    # Save the figure
    plt.savefig(rf'{save_filepath}\{name}{time}.png', bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()