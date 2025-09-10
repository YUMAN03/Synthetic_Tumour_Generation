"""
TUMOR INPAINTING GAN PIPELINE - PART 1: PREPROCESSING
1. Extracts tumor patches from DICOM images
2. Creates masked versions (tumor removed)
3. Prepares data for GAN training
"""

import os
import pydicom
import cv2
import numpy as np
import pandas as pd
from glob import glob
# File paths

DICOM_DIR = "/home/shivam/tumor_deepfake_GAN/unhealthy_scans_raw"
CSV_PATH = "/home/shivam/tumor_deepfake_GAN/dicom_metadata.csv"
OUTPUT_DIR_EXTENDED = "/home/shivam/tumor_deepfake_GAN/extended_tumors"
OUTPUT_DIR_MASKED = "/home/shivam/tumor_deepfake_GAN/masked_extended_tumors"
REGION_METADATA_PATH = "/home/shivam/tumor_deepfake_GAN/data.csv"

def create_directories():
    """Create necessary output directories"""
    os.makedirs(OUTPUT_DIR_EXTENDED, exist_ok=True)
    os.makedirs(OUTPUT_DIR_MASKED, exist_ok=True)

def process_dicom(dicom_file, tumor_border):
    """Extracts tumor patches and creates masked versions"""
    try:
        # Parse tumor coordinates
        tumor_coords = eval(tumor_border)
        tumor_coords = [item[0] if isinstance(item, list) else item for item in tumor_coords]
        
        if len(tumor_coords) % 2 != 0:
            raise ValueError("Tumor coordinates should be pairs of (x,y) values")
            
        x_vals = tumor_coords[0::2]
        y_vals = tumor_coords[1::2]
        tumor_coords = list(zip(x_vals, y_vals))
    except Exception as e:
        print(f"Error parsing tumor coordinates for {dicom_file}: {e}")
        return None

    # Load and process DICOM
    dicom_path = os.path.join(DICOM_DIR, dicom_file)
    if not os.path.exists(dicom_path):
        print(f"DICOM file not found: {dicom_file}")
        return None

    dicom_data = pydicom.dcmread(dicom_path)
    if not hasattr(dicom_data, "PixelRepresentation"):
        dicom_data.add_new((0x0028, 0x0103), 'US', 0)

    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)

    # Calculate tumor bounding box
    xmin, xmax = int(min(x_vals)), int(max(x_vals))
    ymin, ymax = int(min(y_vals)), int(max(y_vals))
    expansion = 28
    xmin = max(0, xmin - expansion)
    xmax = min(image.shape[1], xmax + expansion)
    ymin = max(0, ymin - expansion)
    ymax = min(image.shape[0], ymax + expansion)

    # Extract and mask tumor region
    extended_region = image[ymin:ymax, xmin:xmax]
    masked_region = extended_region.copy()
    
    tumor_xmin, tumor_xmax = min(x_vals) - xmin, max(x_vals) - xmin
    tumor_ymin, tumor_ymax = min(y_vals) - ymin, max(y_vals) - ymin
    masked_region[int(tumor_ymin):int(tumor_ymax), int(tumor_xmin):int(tumor_xmax)] = 0

    # Save patches
    base_name = dicom_file.replace('.dcm', '')
    cv2.imwrite(os.path.join(OUTPUT_DIR_EXTENDED, f"{base_name}.png"), extended_region)
    cv2.imwrite(os.path.join(OUTPUT_DIR_MASKED, f"{base_name}_masked.png"), masked_region)

    return {
        'filename': f"{base_name}.png",
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax
    }

def resize_patches(directory, size=(64, 64)):
    """Resizes all patches in directory to target size"""
    for img_path in glob(os.path.join(directory, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, resized_img)

def run_preprocessing():
    """Execute complete preprocessing pipeline"""
    create_directories()
    metadata_list = []
    df = pd.read_csv(CSV_PATH)
    
    for _, row in df.iterrows():
        result = process_dicom(row['DICOM_File'], row['Tumor_Border'])
        if result:
            metadata_list.append(result)
    
    pd.DataFrame(metadata_list).to_csv(REGION_METADATA_PATH, index=False)
    resize_patches(OUTPUT_DIR_EXTENDED)
    resize_patches(OUTPUT_DIR_MASKED)
    print("Preprocessing completed successfully")

if __name__ == "__main__":
    run_preprocessing()
