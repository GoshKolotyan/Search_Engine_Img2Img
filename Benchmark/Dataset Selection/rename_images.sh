#!/bin/bash

WEB_SITE_NAME="bathdepot"
CATEGORY="Toilet"
IMAGE_DIRECTORY="Images_Bathdepot/Toilet_URLs"  

cd "$IMAGE_DIRECTORY" || exit

COUNTER=1

for file in image_*_page_*.jpg; do
    NEW_FILENAME="${WEB_SITE_NAME}_${CATEGORY}_${COUNTER}.jpg"
    
    mv "$file" "$NEW_FILENAME"
    echo "Renamed: $file -> $NEW_FILENAME"

    ((COUNTER++))
done
