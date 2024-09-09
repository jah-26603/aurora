# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:55:07 2024

@author: JDawg
"""
import tarfile
import glob
import os


def read_tar(parent_dir):
    tar_files = glob.glob(f"{parent_dir}/*.tar")
    
    
    with tarfile.open(tar_files[0], "r") as tar:
        for member in tar.getmembers():
            # Extract only the file name (base name)
            base_name = os.path.basename(member.name)
            
            # Define the target path for saving the file
            target_path = os.path.join(parent_dir, base_name)
            
            # Extract the file content into memory
            file_content = tar.extractfile(member).read()
            
            # Write the file content to the target path
            with open(target_path, "wb") as f:
                f.write(file_content)
            
    
    print("Tar file extraction completed.")
    
    
if __name__ == "__main__":
    parent_dir = r"C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\Aurora_Dates\03_21_2023"
    read_tar(parent_dir)