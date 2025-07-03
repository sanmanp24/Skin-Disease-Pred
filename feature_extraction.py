import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from glob import glob
import os

def extract_features(image):
    # Ensure the image is grayscale
    if len(image.shape) == 3:  # Convert to grayscale if it's in color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # LBP Histogram
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum()
    
    # LAB Color Histogram
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
    color_hist = cv2.calcHist([lab_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    color_hist /= color_hist.sum()
    
    # GLCM Texture Features
    glcm = graycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    glcm_features = [contrast, dissimilarity, homogeneity, asm]
    
    # HOG Features
    hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    
    # Combine all features into a single vector
    features = np.concatenate([lbp_hist, color_hist, glcm_features, hog_features])
    return features

# Paths
input_folder = r'D:\Users\sanma\DermaRX\PreprocessedData'
features = []
labels = []

# Dictionary of disease labels
disease_labels = {
    "Basal Cell Carcinoma (BCC)": 0,
    "Benign Keratosis-like Lesions (BKL)": 1,
    "Melanocytic Nevi (NV)": 2,
    "Melanoma": 3,
}

# Extract features for each image
for disease_folder in os.listdir(input_folder):
    disease_path = os.path.join(input_folder, disease_folder)
    disease_label = disease_labels[disease_folder]
    
    for filepath in glob(f"{disease_path}/*.jpg"):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        feature_vector = extract_features(image)
        features.append(feature_vector)
        labels.append(disease_label)

# Save features and labels
import pickle
with open('extracted_features_best.pkl', 'wb') as f:
    pickle.dump((features, labels), f)