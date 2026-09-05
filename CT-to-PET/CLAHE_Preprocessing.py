# ============================================================
# PET IMAGE CLAHE PREPROCESSING
# ============================================================
!pip install -q opencv-python numpy matplotlib

# ============================================================
# IMPORT LIBRARIES
# ============================================================
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR = "....../PET/"
OUTPUT_DIR = "....../PET_CLAHE/"

LOW_INTENSITY_THRESHOLD = 50
RATIO_THRESHOLD = 5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# ============================================================
# FUNCTIONS
# ============================================================
def compute_histogram(image): return cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)): return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(image)

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FIND ALL PET IMAGES
# ============================================================
extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]

pet_image_paths = []
for ext in extensions:
    pet_image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

pet_image_paths = sorted(pet_image_paths)

print(f"Total PET images found: {len(pet_image_paths)}")

# ============================================================
# CHECK WHETHER PET IMAGES EXIST
# ============================================================
if len(pet_image_paths) == 0:
    raise FileNotFoundError(f"No PET images found in: {INPUT_DIR}")

# ============================================================
# COUNTERS
# ============================================================
clahe_applied_count = 0
clahe_not_applied_count = 0
failed_count = 0

# ============================================================
# PROCESS EVERY PET IMAGE
# ============================================================
for image_path in pet_image_paths:
    file_name = os.path.basename(image_path)

    try:
        PET = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if PET is None:
            print(f"Could not read: {file_name}")
            failed_count += 1
            continue

        # ----------------------------------------------------
        # Count pixels below and above intensity 50
        # ----------------------------------------------------
        low_intensity_count = np.sum(PET < LOW_INTENSITY_THRESHOLD)
        high_intensity_count = np.sum(PET >= LOW_INTENSITY_THRESHOLD)

        # ----------------------------------------------------
        # Calculate low/high intensity ratio
        # ----------------------------------------------------
        intensity_ratio = low_intensity_count / max(high_intensity_count, 1)

        # ----------------------------------------------------
        # Apply CLAHE if low-intensity pixels > 5 ×
        # high-intensity pixels
        # ----------------------------------------------------
        if low_intensity_count > RATIO_THRESHOLD * high_intensity_count:
            PET_processed = apply_clahe(PET,clip_limit=CLAHE_CLIP_LIMIT,grid_size=CLAHE_GRID_SIZE)
            clahe_applied_count += 1
            status = "CLAHE applied"
        else:
            PET_processed = PET.copy()
            clahe_not_applied_count += 1
            status = "CLAHE not applied"

        # ----------------------------------------------------
        # Save processed PET image
        # ----------------------------------------------------
        output_path = os.path.join(OUTPUT_DIR, file_name)

        if not cv2.imwrite(output_path, PET_processed):
            print(f"Failed to save: {file_name}")
            failed_count += 1
            continue

        # ----------------------------------------------------
        # Display processing information
        # ----------------------------------------------------
        print(
            f"{file_name} | "
            f"Low: {low_intensity_count} | "
            f"High: {high_intensity_count} | "
            f"Ratio: {intensity_ratio:.2f} | "
            f"{status}"
        )

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        failed_count += 1

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"Total PET images      : {len(pet_image_paths)}")
print(f"CLAHE applied         : {clahe_applied_count}")
print(f"CLAHE not applied     : {clahe_not_applied_count}")
print(f"Failed images         : {failed_count}")
print(f"Output directory      : {OUTPUT_DIR}")

# ============================================================
# VISUALIZE ONE EXAMPLE
# ============================================================
example_index = 0
example_path = pet_image_paths[example_index]
example_name = os.path.basename(example_path)
original = cv2.imread(example_path,cv2.IMREAD_GRAYSCALE)
processed = cv2.imread(os.path.join(OUTPUT_DIR, example_name),cv2.IMREAD_GRAYSCALE)

# ============================================================
# HISTOGRAMS
# ============================================================
hist_original = compute_histogram(original)
hist_processed = compute_histogram(processed)

# ============================================================
# DISPLAY ORIGINAL / PROCESSED IMAGE AND HISTOGRAM
# ============================================================
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(original, cmap="gray")
plt.title("Original PET Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(processed, cmap="gray")
plt.title("Processed PET Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.plot(hist_original)
plt.xlim([0, 256])
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Original PET Histogram")

plt.subplot(2, 2, 4)
plt.plot(hist_processed)
plt.xlim([0, 256])
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Processed PET Histogram")

plt.tight_layout()
plt.show()