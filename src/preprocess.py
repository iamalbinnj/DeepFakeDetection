# %%
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from mtcnn import MTCNN

# %%
# Paths
DATA_DIR = '../dataset'
OUTPUT_DIR = '../models/preprocessed'
TARGET_SIZE = (299, 299)

# %%
def preprocess_image(image_path):
    print(f"Processing image: {image_path}")  # Debug print
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return None
    
    # Resize image
    if os.path.getsize(image_path) > 9 * 1024:  # Resize if image size is more than 9KB
        image = cv2.resize(image, TARGET_SIZE)
    
    # Detect face
    detector = MTCNN()
    faces = detector.detect_faces(image)
    
    if faces:
        # Crop and mark face
        x, y, width, height = faces[0]['box']
        face = image[y:y+height, x:x+width]
        image = cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), 2)
    
    # Normalize
    image = image.astype("float32") / 255.0
    return image

# %%
def preprocess_data(data_dir):
    data = []
    labels = []
    for category in ['Fake', 'Real']:  # Ensure these are the correct category names
        folder_path = os.path.normpath(os.path.join(data_dir, category))
        if not os.path.exists(folder_path):
            print(f"Warning: Folder does not exist: {folder_path}")
            continue
        print(f"Processing folder: {folder_path}")  # Debug print
        image_files = os.listdir(folder_path)
        if not image_files:
            print(f"Warning: No images found in folder: {folder_path}")
        for img in image_files:
            img_path = os.path.normpath(os.path.join(folder_path, img))
            image = preprocess_image(img_path)
            if image is not None:
                data.append(image)
                labels.append(0 if category == 'Fake' else 1)
    
    return np.array(data), np.array(labels)

# %%
# Create the output directory if it does not exist
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")  # Debug print
    except Exception as e:
        print(f"Error creating directory {OUTPUT_DIR}: {e}")
        raise

# %%
# Preprocess Train dataset
try:
    print("Starting preprocessing for Train dataset...")
    train_data, train_labels = preprocess_data(os.path.join(DATA_DIR, 'Train'))
    print("Train dataset preprocessing complete.")
    # Save preprocessed data
    np.save(os.path.join(OUTPUT_DIR, 'train_data.npy'), train_data)
    np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), train_labels)
    
    print("Data saved successfully.")  # Debug print
except Exception as e:
    print(f"Error during data processing or saving: {e}")

# %%
# Preprocess Test dataset
try:
    print("Starting preprocessing for Test dataset...")
    test_data, test_labels = preprocess_data(os.path.join(DATA_DIR, 'Test'))
    print("Test dataset preprocessing complete.")
    
    # Save preprocessed data
    np.save(os.path.join(OUTPUT_DIR, 'test_data.npy'), test_data)
    np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), test_labels)
    
    print("Data saved successfully.")  # Debug print
except Exception as e:
    print(f"Error during data processing or saving: {e}")

# %%
# Preprocess Valid dataset
try:
    print("Starting preprocessing for Validation dataset...")
    valid_data, valid_labels = preprocess_data(os.path.join(DATA_DIR, 'Validation'))
    print("Validation dataset preprocessing complete.")
    
    # Save preprocessed data
    np.save(os.path.join(OUTPUT_DIR, 'valid_data.npy'), valid_data)
    np.save(os.path.join(OUTPUT_DIR, 'valid_labels.npy'), valid_labels)
    
    print("Data saved successfully.")  # Debug print
except Exception as e:
    print(f"Error during data processing or saving: {e}")

# %%
# Load the preprocessed data
train_data = np.load(os.path.join(OUTPUT_DIR, 'train_data.npy'))
train_labels = np.load(os.path.join(OUTPUT_DIR, 'train_labels.npy'))

print(f"Train data shape: {train_data.shape}")
print(f"Train labels shape: {train_labels.shape}")

# Repeat for test and validation data
test_data = np.load(os.path.join(OUTPUT_DIR, 'test_data.npy'))
test_labels = np.load(os.path.join(OUTPUT_DIR, 'test_labels.npy'))

valid_data = np.load(os.path.join(OUTPUT_DIR, 'valid_data.npy'))
valid_labels = np.load(os.path.join(OUTPUT_DIR, 'valid_labels.npy'))


