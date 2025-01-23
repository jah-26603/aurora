#!/bin/bash


BASE_URL="https://spdf.gsfc.nasa.gov/pub/data/gold/level1c/2020/" #url of website to download from
OUTPUT_DIR="D:/gold_level1c_2020_every_7th/" #parent directory output
mkdir -p "$OUTPUT_DIR" #create dir if DNE

# Creates a list of all folders in cwd
curl -s "$BASE_URL" | grep -oP '(?<=href=")[^"]*/' > folder_list.txt

# Creates a new list that selects every 7th folder
awk 'NR % 7 == 0' folder_list.txt > every_7th_folder.txt 

# Download each 7th day folder and only corresponding DAY.nc files
while read -r folder; do
    echo "Processing folder: $folder"
    
    # Create a local folder structure for the day folder and creates a file list of day scans
    LOCAL_FOLDER="$OUTPUT_DIR$folder"
    mkdir -p "$LOCAL_FOLDER"
    curl -s "${BASE_URL}${folder}" | grep -oP '(?<=href=")[^"]*' | grep "DAY" > files_in_folder.txt 
    
    # Downloads all relevant file
    while read -r file; do
        LOCAL_FILE="${LOCAL_FOLDER}${file}"
        if [ -f "$LOCAL_FILE" ]; then
            echo "File ${file} already exists. Skipping."
        else
        echo "Downloading ${file} from ${folder}..."
        #If don't have aria2c, can download from github and add to PATH,
        #or replace w/ curl -s... from line 21 
        aria2c -x 16 -s 16 -d "$LOCAL_FOLDER" -o "$file" "${BASE_URL}${folder}${file}"
        fi
    done < files_in_folder.txt
    
done < every_7th_folder.txt

echo "Download complete. Files stored in $OUTPUT_DIR."
