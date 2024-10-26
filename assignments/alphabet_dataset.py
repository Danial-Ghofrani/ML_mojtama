import os
import shutil
import cv2
import numpy as np

# Define image size and dataset directory
IMAGE_SIZE = (64, 64)
DATASET_DIR = r"D:\programming\Machine_learning_Ostad_Mesbah\assignments\alphabet_processed_dataset"
PHOTOS_DIR = r"D:\programming\Machine_learning_Ostad_Mesbah\assignments\alphabet_raw_dataset_1"

# Clear previous dataset if exists
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
os.mkdir(DATASET_DIR)

# Process images in the specified 'photos' folder
photo_files = [f for f in os.listdir(PHOTOS_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]

for photo_file in photo_files:
    # Load the photo in grayscale
    image = cv2.imread(os.path.join(PHOTOS_DIR, photo_file), cv2.IMREAD_GRAYSCALE)

    # Resize to 64x64
    image = cv2.resize(image, IMAGE_SIZE)

    # Apply binary threshold to make letters white and background black
    _, image = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY_INV)

    # Save the processed image with the same filename
    image_path = os.path.join(DATASET_DIR, photo_file)
    cv2.imwrite(image_path, image)

# Optionally save as a .npz file if needed later
np.savez_compressed("alphabet_dataset.npz",
                    images=[cv2.imread(os.path.join(DATASET_DIR, f), cv2.IMREAD_GRAYSCALE) for f in photo_files])
