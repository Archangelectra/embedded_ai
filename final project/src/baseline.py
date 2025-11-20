import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# CONFIGURATION
DATASET_PATH = "data" 
NUM_IDENTITIES = 20          # SVC is slow; start with a small number of classes
IMAGES_PER_ID = 20           # use a subset of images per person

def load_digiface_subset(path, num_ids, imgs_per_id):
    X = []
    y = []
    
    # Get list of identity folders
    identities = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])[:num_ids]
    
    print(f"Loading {num_ids} identities...")
    
    for label, identity_id in enumerate(identities):
        folder_path = os.path.join(path, identity_id)
        image_files = os.listdir(folder_path)[:imgs_per_id]
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            # 1. load as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # 2. flatten image
                X.append(img.flatten())
                y.append(label) # label is the index (0 to 19)

    return np.array(X), np.array(y), identities

# 1. load data
X, y, target_names = load_digiface_subset(DATASET_PATH, NUM_IDENTITIES, IMAGES_PER_ID)
print(f"Data Shape: {X.shape}") # should be (400, 12544)

# 2. split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. build the pipeline
# standard scaler: normalizes pixel intensity
# PCA: reduces 12k pixels to 150 "eigenface" components
# SVC: the classifier itself
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced', C=1000, gamma=0.005)

model = make_pipeline(StandardScaler(), pca, svc)

# 4. Train (Fit)
print("Training SVC (this may take a moment)...")
model.fit(X_train, y_train)

# 5. Evaluate
print("\n--- Baseline Results ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))