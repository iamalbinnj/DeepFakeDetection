# %%
import os
import cv2
from mtcnn import MTCNN

# %%
# Paths
DATA_DIR = os.path.abspath('E:\\deepfake_detector\\dataset')  # Path to your dataset
RESIZED_DIR = os.path.abspath('E:\\deepfake_detector\\dataset\\mapped\\resized')  # Path to save resized images
TARGET_SIZE = (299, 299)

# %%
def process_image(image_path, output_path, label):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image {image_path}")
        
        # Resize the image if it's larger than 9KB
        if os.path.getsize(image_path) > 9 * 1024:
            image = cv2.resize(image, TARGET_SIZE)
        
        # Noise Reduction (using Gaussian Blur)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Contrast Enhancement (using CLAHE)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Edge Detection (using Canny)
        edges = cv2.Canny(image, 100, 200)
        
        # Detect faces using MTCNN
        detector = MTCNN()
        faces = detector.detect_faces(image)
        
        # If a face is detected, mark it with a rectangle
        if faces:
            x, y, width, height = faces[0]['box']
            image = cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
        
        # Add the label text ("Fake" or "Real") to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Save the processed image
        cv2.imwrite(output_path, image)
        return output_path
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# %%
def load_images(data_dir, output_dir, max_images=None):
    data = []
    labels = []
    for category in ['Fake', 'Real']:
        category_path = os.path.join(data_dir, category)
        output_category_path = os.path.join(output_dir, category)
        
        if not os.path.exists(category_path):
            continue
        
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)
        
        total_images = len(os.listdir(category_path))
        count = 0
        for img in os.listdir(category_path):
            if max_images and count >= max_images:
                break
            
            img_path = os.path.join(category_path, img)
            output_path = os.path.join(output_category_path, img)
            processed_image_path = process_image(img_path, output_path, label=category)
            
            if processed_image_path:
                data.append(processed_image_path)
                labels.append(0 if category == 'Fake' else 1)
                count += 1
            
            # Print progress
            print(f"Processed {count}/{min(max_images or total_images, total_images)} images in '{category}' category")

    return data, labels

# %%
# Ensure the output directory exists
if not os.path.exists(RESIZED_DIR):
    os.makedirs(RESIZED_DIR)

# Process Train and Validation datasets only
try:
    print(f"Processing Train dataset...")
    train_data, train_labels = load_images(os.path.join(DATA_DIR, 'Train'), os.path.join(RESIZED_DIR, 'Train'), max_images=10000)
    
    print(f"Processing Validation dataset...")
    valid_data, valid_labels = load_images(os.path.join(DATA_DIR, 'Validation'), os.path.join(RESIZED_DIR, 'Validation'))
    
    # Print some debug information
    print(f"Processed Train data: {len(train_data)} images")
    print(f"Processed Validation data: {len(valid_data)} images")

except Exception as e:
    print(f"Error during dataset processing: {e}")


