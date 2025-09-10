"""
TUMOR INPAINTING GAN PIPELINE - PART 1: PREPROCESSING & FAKE AUGMENTATION
1. Extracts tumor patches from DICOM images
2. Creates masked versions (tumor removed)
3. Prepares data for GAN training
4. Inserts GAN-generated fake tumors within central box at random location
"""

import os
import pydicom
import cv2
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from training_code_unet_pix2pix import UNetGenerator

# File paths
DICOM_DIR = "/home/shivam/tumor_deepfake_GAN/unhealthy_scans_raw"
CSV_PATH = "/home/shivam/tumor_deepfake_GAN/dicom_metadata.csv"
OUTPUT_DIR_EXTENDED = "/home/shivam/tumor_deepfake_GAN/extended_tumors"
OUTPUT_DIR_MASKED = "/home/shivam/tumor_deepfake_GAN/masked_extended_tumors"
OUTPUT_DIR_FAKE = "/home/shivam/tumor_deepfake_GAN/fake_augmented_images_unet_pix2pix_2"
REGION_METADATA_PATH = "/home/shivam/tumor_deepfake_GAN/data.csv"
FAKE_REGION_METADATA_PATH = "/home/shivam/tumor_deepfake_GAN/data_2.csv"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_directories():
    os.makedirs(OUTPUT_DIR_EXTENDED, exist_ok=True)
    os.makedirs(OUTPUT_DIR_MASKED, exist_ok=True)
    os.makedirs(OUTPUT_DIR_FAKE, exist_ok=True)

def process_dicom(dicom_file, tumor_border):
    try:
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

    dicom_path = os.path.join(DICOM_DIR, dicom_file)
    if not os.path.exists(dicom_path):
        print(f"DICOM file not found: {dicom_file}")
        return None

    dicom_data = pydicom.dcmread(dicom_path)
    if not hasattr(dicom_data, "PixelRepresentation"):
        dicom_data.add_new((0x0028, 0x0103), 'US', 0)

    image = dicom_data.pixel_array
    image = apply_windowing(dicom_data)  # or ds in insert_fake_tumor

    image = image.astype(np.uint8)

    xmin, xmax = int(min(x_vals)), int(max(x_vals))
    ymin, ymax = int(min(y_vals)), int(max(y_vals))
    expansion = 28
    xmin = max(0, xmin - expansion)
    xmax = min(image.shape[1], xmax + expansion)
    ymin = max(0, ymin - expansion)
    ymax = min(image.shape[0], ymax + expansion)

    extended_region = image[ymin:ymax, xmin:xmax]
    masked_region = extended_region.copy()
    tumor_xmin, tumor_xmax = min(x_vals) - xmin, max(x_vals) - xmin
    tumor_ymin, tumor_ymax = min(y_vals) - ymin, max(y_vals) - ymin
    masked_region[int(tumor_ymin):int(tumor_ymax), int(tumor_xmin):int(tumor_xmax)] = 0

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
    for img_path in glob(os.path.join(directory, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, resized_img)

import os
import numpy as np
import pydicom
import cv2
from PIL import Image
import torch
from torchvision import transforms

def insert_fake_tumor(dicom_file, real_bbox, generator, device):
    """
    Inserts a GAN-generated fake tumor patch into a fixed central region of the scan.
    Patch size and mask size are randomized within specified ranges.
    The central region is masked and inpainted using the generator.
    Blending is applied for smooth insertion into the scan.
    """

    # Load DICOM file
    dicom_path = os.path.join(DICOM_DIR, dicom_file)
    ds = pydicom.dcmread(dicom_path)

    # Ensure PixelRepresentation exists for valid image extraction
    if not hasattr(ds, "PixelRepresentation"):
        ds.add_new((0x0028, 0x0103), 'US', 0)

    # Convert DICOM image to normalized 8-bit grayscale
    image = ds.pixel_array.astype(np.float32)
    image = apply_windowing(ds)  # or ds in insert_fake_tumor

    image = image.astype(np.uint8)

    H, W = image.shape
    xmin_real, xmax_real, ymin_real, ymax_real = real_bbox  # Bounding box of real tumor

    # Randomize patch size and corresponding mask size
    patch_size = np.random.choice([64, 72, 80, 88, 96])
    mask_frac = np.random.uniform(0.3, 0.7)
    mask_size = int(patch_size * mask_frac)

    pad = 5
    max_attempts = 20

    # Define central box region for patch placement
    center_h, center_w = H // 2, W // 2
    margin = 100  # Size of half the central box edge

    box_top = max(pad, center_h - margin)
    box_bottom = min(H - patch_size - pad, center_h + margin)
    box_left = max(pad, center_w - margin)
    box_right = min(W - patch_size - pad, center_w + margin)

    # Randomly find a non-overlapping position within the central box
    for _ in range(max_attempts):
        x_rand = np.random.randint(box_left, box_right)
        y_rand = np.random.randint(box_top, box_bottom)

        # Ensure patch does not overlap real tumor
        if not (xmin_real - pad < x_rand < xmax_real + pad and ymin_real - pad < y_rand < ymax_real + pad):
            break
    else:
        print(f"No valid position found in center box for {dicom_file}")
        return None

    # Extract patch from image
    patch = image[y_rand:y_rand + patch_size, x_rand:x_rand + patch_size].copy()

    # Apply centered square mask to the patch
    cx, cy = patch_size // 2, patch_size // 2
    half_mask = mask_size // 2
    patch[cx - half_mask:cx + half_mask, cy - half_mask:cy + half_mask] = 0  # Black mask

    # Resize patch to 64x64 as input to generator
    patch_input = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])(Image.fromarray(patch_input)).unsqueeze(0).to(device)

    # Predict inpainted patch using generator
    with torch.no_grad():
        fake_patch = generator(input_tensor).squeeze().cpu().numpy()
        fake_patch = (fake_patch * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    # Resize generated patch back to original patch size
    fake_patch_resized = cv2.resize(fake_patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

    # Create smooth blending mask using cosine window
    cosine_window = np.outer(np.hanning(patch_size), np.hanning(patch_size))
    blending_mask = cosine_window.astype(np.float32)

    # Blend the fake patch into the original image region
    original_region = image[y_rand:y_rand + patch_size, x_rand:x_rand + patch_size].astype(np.float32)
    blended = (blending_mask * fake_patch_resized + (1 - blending_mask) * original_region).astype(np.uint8)

    # Update the original image with the blended patch
    image[y_rand:y_rand + patch_size, x_rand:x_rand + patch_size] = blended

    # Save final image
    out_path = os.path.join(OUTPUT_DIR_FAKE, dicom_file.replace('.dcm', '_fake.png'))
    cv2.imwrite(out_path, image)

    # Return annotation info
    return {
        'filename': os.path.basename(out_path),
        'xmin': x_rand,
        'xmax': x_rand + patch_size,
        'ymin': y_rand,
        'ymax': y_rand + patch_size,
        'patch_size': patch_size,
        'mask_size': mask_size
    }

def apply_windowing(dicom):
    """Apply DICOM windowing using WindowCenter and WindowWidth"""
    img = dicom.pixel_array.astype(np.float32)
    
    # Get window center and width
    window_center = dicom.get('WindowCenter', None)
    window_width = dicom.get('WindowWidth', None)
    
    # If multiple values are provided, take the first
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = window_center[0]
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = window_width[0]

    if window_center and window_width:
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        img = np.clip(img, min_val, max_val)
        img = ((img - min_val) / (max_val - min_val)) * 255.0
    else:
        # Fall back to min-max normalization (not ideal)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0

    return img.astype(np.uint8)




def run_preprocessing(generator=None):
    create_directories()
    metadata_list = []
    fake_metadata_list = []
    df = pd.read_csv(CSV_PATH)

    for _, row in df.iterrows():
        result = process_dicom(row['DICOM_File'], row['Tumor_Border'])
        if result:
            metadata_list.append(result)
            if generator:
                fake_result = insert_fake_tumor(row['DICOM_File'],
                                                (result['xmin'], result['xmax'], result['ymin'], result['ymax']),
                                                generator, device)
                if fake_result:
                    fake_metadata_list.append(fake_result)

    pd.DataFrame(metadata_list).to_csv(REGION_METADATA_PATH, index=False)
    resize_patches(OUTPUT_DIR_EXTENDED)
    resize_patches(OUTPUT_DIR_MASKED)

    if generator and fake_metadata_list:
        pd.DataFrame(fake_metadata_list).to_csv(FAKE_REGION_METADATA_PATH, index=False)

    print("Preprocessing and augmentation completed successfully")

if __name__ == "__main__":
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load("/home/shivam/tumor_deepfake_GAN/generator_pix2pix_unet.pth", map_location=device))
    generator.eval()
    run_preprocessing(generator=generator)
