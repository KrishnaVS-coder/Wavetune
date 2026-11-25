import os
import cv2
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")

IMG_SIZE = 64

data = []
labels = []

print("Starting preprocessing...\n")

for label_name in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label_name)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing folder: {label_name}")

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            data.append(img)
            labels.append(label_name)

        except:
            print("Skipping:", img_name)

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

print("\nSaving processed data...")

np.save(os.path.join(MODEL_PATH, "X.npy"), X)
np.save(os.path.join(MODEL_PATH, "y.npy"), y)

print("Saved successfully!")
print("Total images:", len(X))
