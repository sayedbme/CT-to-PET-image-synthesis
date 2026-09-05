import os
import glob
import numpy as np
from ultralytics import YOLO
from skimage import io

# Constants
BASE_DIR = ".../dataset/"
CT_MODEL_PATH = ".../ct_best.pt"
PET_MODEL_PATH = ".../pet_best.pt"
CONFIDENCE_THRESHOLD = 0.10

# Load YOLO models
ct_model = YOLO(CT_MODEL_PATH)
pet_model = YOLO(PET_MODEL_PATH)

# Process each patient's folder
patient_folders = sorted(glob.glob(os.path.join(BASE_DIR, "*")))

for patient_folder in patient_folders:
    print(f"Processing patient folder: {patient_folder}")

    # Directories for CT and PET
    ct_dir = os.path.join(patient_folder, "CT")
    pet_dir = os.path.join(patient_folder, "PET")

    # Process PET images
    pet_images = sorted(glob.glob(os.path.join(pet_dir, "*.png")))
    pet_results = pet_model.predict(source=pet_dir, conf=CONFIDENCE_THRESHOLD)

    for i, img_path in enumerate(pet_images):
        try:
            # Check for valid bounding boxes
            if i < len(pet_results) and pet_results[i].boxes.xyxy is not None and pet_results[i].boxes.xyxy.shape[0] > 0:
                bbox_raw = pet_results[i].boxes.xyxy[0].numpy().astype(int)  # Get first bounding box and convert to int
                original_image = io.imread(img_path)
                
                # Crop the image using bounding box coordinates
                cropped_image = original_image[bbox_raw[1]:bbox_raw[3], bbox_raw[0]:bbox_raw[2]]

                # Save cropped image, replacing the existing file
                io.imsave(img_path, cropped_image)
            else:
                print(f"No bounding boxes detected for PET image {img_path}, skipping.")
        except Exception as e:
            print(f"Error processing PET image {img_path}: {e}, skipping.")

    # Process CT images
    ct_images = sorted(glob.glob(os.path.join(ct_dir, "*.png")))
    ct_results = ct_model.predict(source=ct_dir, conf=CONFIDENCE_THRESHOLD)

    for i, img_path in enumerate(ct_images):
        try:
            # Check for valid bounding boxes
            if i < len(ct_results) and ct_results[i].boxes.xyxy is not None and ct_results[i].boxes.xyxy.shape[0] > 0:
                bbox_raw = ct_results[i].boxes.xyxy[0].numpy().astype(int)  # Get first bounding box and convert to int
                original_image = io.imread(img_path)

                # Crop the image using bounding box coordinates
                cropped_image = original_image[bbox_raw[1]:bbox_raw[3], bbox_raw[0]:bbox_raw[2]]

                # Save cropped image, replacing the existing file
                io.imsave(img_path, cropped_image)
            else:
                print(f"No bounding boxes detected for CT image {img_path}, skipping.")
        except Exception as e:
            print(f"Error processing CT image {img_path}: {e}, skipping.")

print("Batch detection and cropping completed.")
