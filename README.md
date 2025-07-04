# Skin-Lesion-Classification
**Skin Lesion Classification Using Traditional Machine Learning**

This project implements a robust machine learning pipeline for the classification of four common skin lesion types: basal cell carcinoma, benign keratosis-like lesions, melanocytic nevi, and melanoma. The system leverages traditional ensemble models—Random Forest and Gradient Boosting—combined with comprehensive image preprocessing, augmentation, and feature extraction techniques to achieve accurate and interpretable results.

Key Features

Data Preprocessing & Augmentation: Standardizes dermoscopic images through grayscale conversion, adaptive thresholding, sharpening, resizing, and padding. Augmentation techniques (rotation, flipping, scaling, color jittering, Gaussian blur, gamma correction) address class imbalance and improve model generalizability.

Feature Extraction: Utilizes Local Binary Patterns (LBP), LAB color histograms, Gray-Level Co-occurrence Matrix (GLCM), and Histogram of Oriented Gradients (HOG) to capture essential texture, color, and structural features from images.

Ensemble Classification: Combines Random Forest and Gradient Boosting classifiers, with hyperparameter optimization and ensemble voting strategies to maximize predictive performance.

Evaluation: Employs precision, recall, F1-score, and confusion matrix analyses to rigorously assess model accuracy across all lesion classes.

User Interface: Includes a Tkinter-based GUI for real-time image upload and instant lesion classification with probability outputs.

Why Traditional ML?
This approach offers a practical, interpretable alternative to deep learning, requiring fewer computational resources and providing transparent decision support—ideal for deployment in resource-constrained clinical settings.

Repository Contents

Source code for data preprocessing, feature extraction, model training, and evaluation

Scripts for hyperparameter tuning and ensemble integration

GUI application for user-friendly predictions

Documentation detailing methods, dataset characteristics, and usage instructions

Ideal for: Researchers, clinicians, and developers interested in accessible, explainable AI solutions for dermatological diagnostics
![image](https://github.com/user-attachments/assets/fc39065f-9e77-4bd3-b9a5-9b278c9d5ac0)
![image](https://github.com/user-attachments/assets/1656dd1f-8489-434d-8d18-c4768e4172bc)
![image](https://github.com/user-attachments/assets/2891162e-1ef0-4d52-a26d-163b291a748f)
![image](https://github.com/user-attachments/assets/1fdc51a3-0c79-4021-a210-e4752f0cd027)
![image](https://github.com/user-attachments/assets/35d36fe0-e356-4515-8561-02fc99f12ac2)
![image](https://github.com/user-attachments/assets/4201553b-9a11-4d89-a2fe-d42f4cf40d25)
![image](https://github.com/user-attachments/assets/a8830870-f81c-4738-b8d3-0a993ea72203)


