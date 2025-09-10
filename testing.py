"""
TUMOR INPAINTING GAN PIPELINE - PART 3: TESTING
1. Loads trained generator model
2. Reconstructs tumors in masked patches
3. Saves output as PNG images
"""

import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import torch
from torchvision import transforms
from training_code_1 import Generator
# Reuse paths from preprocessing
DICOM_DIR = "/home/shivam/tumor_deepfake_GAN/unhealthy_scans_raw"
OUTPUT_DIR_MASKED = "/home/shivam/tumor_deepfake_GAN/masked_extended_tumors"
REGION_METADATA_PATH = "/home/shivam/tumor_deepfake_GAN/data.csv"
MODEL_PATH = "/home/shivam/tumor_deepfake_GAN/generator.pth"

def reconstruct_as_png(output_dir):
    """Reconstruct tumors and save as PNG using trained model"""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(MODEL_PATH))
    generator.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load metadata
    metadata = pd.read_csv(REGION_METADATA_PATH)
    
    for _, row in metadata.iterrows():
        # Load original DICOM
        dicom_path = os.path.join(DICOM_DIR, row['filename'].replace('.png', '.dcm'))
        ds = pydicom.dcmread(dicom_path)
        if not hasattr(ds, "PixelRepresentation"):
            print(f"⚠️ Warning: (0028,0103) 'Pixel Representation' missing for dicom_file. Assuming unsigned data.")
            ds.add_new((0x0028, 0x0103), 'US', 0)
        img = ds.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)

        # Generate tumor
        masked_path = os.path.join(OUTPUT_DIR_MASKED, row['filename'].replace('.png', '_masked.png'))
        masked_img = Image.open(masked_path).convert("L")
        masked_tensor = transform(masked_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated = generator(masked_tensor).squeeze().cpu().numpy()
            generated = (generated * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

        # Resize and blend
        xmin, xmax, ymin, ymax = row['xmin'], row['xmax'], row['ymin'], row['ymax']
        patch_height, patch_width = ymax - ymin, xmax - xmin
        resized_tumor = cv2.resize(generated, (patch_width, patch_height))
        img[ymin:ymax, xmin:xmax] = resized_tumor

        # Save PNG
        output_path = os.path.join(output_dir, os.path.basename(dicom_path).replace('.dcm', '.png'))
        cv2.imwrite(output_path, img)
        print(f"Saved reconstructed image: {output_path}")

def run_testing():
    """Execute testing pipeline"""
    OUTPUT_PNG_DIR = "/home/shivam/tumor_deepfake_GAN/reconstructed_images"
    reconstruct_as_png(OUTPUT_PNG_DIR)
    print("Testing and reconstruction completed")

if __name__ == "__main__":
    run_testing()
