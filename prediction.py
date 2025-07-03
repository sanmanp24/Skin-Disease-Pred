import joblib
import cv2
import numpy as np
from tkinter import filedialog, Tk, Button, Label, Toplevel
from skimage.feature import local_binary_pattern
from PIL import Image, ImageTk  # Import PIL for image display

# Load trained model using joblib
with open('trained_model_nice.pkl', 'rb') as f:
    model = joblib.load(f)

# Define labels for prediction output
disease_labels = [
    "Basal Cell Carcinoma (BCC)", "Benign Keratosis-like Lesions (BKL)", 
    "Melanocytic Nevi (NV)", "Melanoma"
]

def preprocess_image(image):
    # Convert image to grayscale if it was done in training
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the expected size used during training
    resized = cv2.resize(gray, (128, 128))
    
    # Normalize pixel values if done in training
    normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def extract_features(image):
    # Local Binary Pattern (LBP) for texture analysis
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    
    # Normalize histogram
    hist = hist.astype("float")
    hist /= hist.sum() if hist.sum() > 0 else 1  # Avoid division by zero
    
    # Compute color histogram for grayscale (if grayscale was used in training)
    color_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    color_hist = color_hist.ravel() / color_hist.sum() if color_hist.sum() > 0 else 1  # Normalize color histogram

    # Concatenate the texture (LBP) histogram and color histogram
    features = np.concatenate([hist, color_hist])
    
    # Check if feature vector needs padding or trimming
    expected_length = 574  # Update this to match the number of features used during training
    if len(features) < expected_length:
        # Pad with zeros if feature vector is shorter
        features = np.pad(features, (0, expected_length - len(features)), mode='maximum')
    elif len(features) > expected_length:
        # Trim if feature vector is longer
        features = features[:expected_length]
    
    # Debug print for feature vector shape
    print("Feature vector shape (prediction):", features.shape)
    
    return features

def classify_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        # Load and preprocess the image
        image = cv2.imread(filepath)
        preprocessed_image = preprocess_image(image)
        features = extract_features(preprocessed_image)

        # Reshape to match input format of the model
        features = features.reshape(1, -1)

        # Make a prediction and get probabilities
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]  # Get prediction probabilities

        # Display the selected image in the Tkinter window
        display_image(image, disease_labels[prediction], probabilities)

def display_image(image, prediction, probabilities):
    # Convert the image from OpenCV (BGR) to PIL format (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_image = pil_image.resize((300, 300), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS

    # Create a Toplevel window to display the image and prediction
    top = Toplevel(root)
    top.title("Prediction Result")

    # Display image in the Toplevel window
    img_display = ImageTk.PhotoImage(pil_image)
    img_label = Label(top, image=img_display)
    img_label.image = img_display  # Keep a reference to avoid garbage collection
    img_label.pack()

    # Display prediction and formatted probabilities
    prediction_text = f"Predicted Disease: {prediction}"
    probabilities_text = "\n".join([f"{disease_labels[i]}: {prob:.2%}" for i, prob in enumerate(probabilities)])

    # Prediction Label
    result_label = Label(top, text=prediction_text, font=("Helvetica", 14, "bold"), fg="blue")
    result_label.pack(pady=10)

    # Probabilities Label
    prob_label = Label(top, text=f"Prediction Probabilities:\n{probabilities_text}", font=("Helvetica", 12))
    prob_label.pack(pady=5)

# Tkinter GUI setup
root = Tk()
root.title("Skin Disease Classifier")

Button(root, text="Classify Image", command=classify_image).pack(pady=20)

# Start the Tkinter event loop
root.mainloop()
