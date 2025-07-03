import cv2
import os
import random
import numpy as np
from glob import glob

# Define directories
input_folder = r'D:\Users\sanma\DermaRX\Data'
output_folder = r'D:\Users\sanma\DermaRX\AugmentedData'
os.makedirs(output_folder, exist_ok=True)

def augment_image(image):
    # Rotation
    rotation_options = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    rotation_choice = random.choice(rotation_options)
    if rotation_choice is not None:
        image = cv2.rotate(image, rotation_choice)
    
    # Flip
    image = cv2.flip(image, random.choice([-1, 0, 1]))
    
    # Brightness
    brightness = random.uniform(0.7, 1.3)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # Contrast adjustment
    contrast = random.uniform(0.7, 1.5)
    image = cv2.convertScaleAbs(image, alpha=contrast)
    
    # Gamma correction
    gamma = random.uniform(0.8, 1.2)
    lookUpTable = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, lookUpTable)
    
    # Gaussian blur
    if random.choice([True, False]):
        image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image

for disease_folder in os.listdir(input_folder):
    disease_path = os.path.join(input_folder, disease_folder)
    output_disease_folder = os.path.join(output_folder, disease_folder)
    os.makedirs(output_disease_folder, exist_ok=True)

    for filepath in glob(f"{disease_path}/*.jpg"):
        image = cv2.imread(filepath)
        for i in range(10):
            augmented_image = augment_image(image)
            filename = os.path.join(output_disease_folder, f"{os.path.basename(filepath).split('.')[0]}_aug_{i}.jpg")
            cv2.imwrite(filename, augmented_image)
