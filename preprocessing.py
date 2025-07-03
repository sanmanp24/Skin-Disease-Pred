import cv2
import os
from glob import glob
import numpy as np
input_folder = r'D:\Users\sanma\DermaRX\AugmentedData'
output_folder = r'D:\Users\sanma\DermaRX\PreprocessedData'
os.makedirs(output_folder, exist_ok=True)

def preprocess_image(image, target_size=(128, 128)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(adaptive_thresh, -1, kernel)
    
    # Resize with aspect ratio
    h, w = sharpened.shape
    if h > w:
        new_h, new_w = target_size[0], int(target_size[1] * (w / h))
    else:
        new_h, new_w = int(target_size[0] * (h / w)), target_size[1]
    
    resized = cv2.resize(sharpened, (new_w, new_h))
    padded_image = cv2.copyMakeBorder(resized, (target_size[0] - new_h) // 2, (target_size[0] - new_h + 1) // 2,
                                      (target_size[1] - new_w) // 2, (target_size[1] - new_w + 1) // 2, 
                                      cv2.BORDER_CONSTANT, value=0)
    return padded_image

for disease_folder in os.listdir(input_folder):
    disease_path = os.path.join(input_folder, disease_folder)
    output_disease_folder = os.path.join(output_folder, disease_folder)
    os.makedirs(output_disease_folder, exist_ok=True)

    for filepath in glob(f"{disease_path}/*.jpg"):
        image = cv2.imread(filepath)
        preprocessed_image = preprocess_image(image)
        filename = os.path.join(output_disease_folder, os.path.basename(filepath))
        cv2.imwrite(filename, preprocessed_image)
