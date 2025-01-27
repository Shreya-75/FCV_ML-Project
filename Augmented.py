import os
import cv2
import numpy as np

# Constants
SOURCE_DIR = 'data'  # Directory containing the original dataset
TARGET_DIR = 'data00'  # Directory to save augmented images
TARGET_IMAGE_COUNT = 1000  # Total number of images per folder

# Create target directory
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)  # Create main target directory if it does not exist


# Function to perform data augmentation
def augment_image(image):
    # Randomly apply a series of transformations
    # 1. Brightness adjustment
    brightness_variation = np.random.uniform(0.7, 1.3)  # Variation factor
    image = cv2.convertScaleAbs(image, alpha=brightness_variation, beta=0)  # Adjust brightness

    # 2. Color variation
    color_variation = np.random.uniform(0.7, 1.3)  # Variation factor
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[..., 1] = np.clip(image[..., 1] * color_variation, 0, 255)  # Adjust saturation
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # 3. Random flipping
    if np.random.rand() > 0.5:  # 50% chance to flip
        image = cv2.flip(image, 1)  # Horizontal flip

    # 4. Random rotation
    angle = np.random.randint(-15, 15)  # Rotate between -15 to 15 degrees
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    return image


# Process each folder and perform augmentation
for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    if os.path.isdir(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'jpeg'))]
        total_images = len(image_files)

        # Create the same subfolder in the target directory
        target_folder = os.path.join(TARGET_DIR, folder)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)  # Create subfolder in data1 if it does not exist

        # Augment images to reach exactly 1000
        count = 0
        while count < TARGET_IMAGE_COUNT:
            image_file = image_files[count % total_images]  # Loop through existing images
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            # Augment the image
            augmented_image = augment_image(image)

            # Save the augmented image (overwrite if same name)
            target_image_path = os.path.join(target_folder, image_file)
            cv2.imwrite(target_image_path, augmented_image)

            count += 1

print("Data augmentation completed with exactly 1000 images per folder!")
