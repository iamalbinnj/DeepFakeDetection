import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

def preprocess_image(image_path, image_size=(224, 224)):
    """Preprocesses an image for deep learning."""
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32 and scale to [0, 1]
        img = tf.keras.applications.xception.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def visualize_images(images, titles, rows=2, cols=4, save_path=None):
    """Visualizes a set of images with corresponding titles."""
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis('off')  # Hide axis labels
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def load_dataset(dataset_dir, split, batch_size=32, limit=None):
    """Loads a dataset from a directory structure."""
    image_paths = [os.path.join(dataset_dir, split, label, filename)
                   for label in ['Fake', 'Real']
                   for filename in os.listdir(os.path.join(dataset_dir, split, label))]
    labels = [1 if 'Fake' in path else 0 for path in image_paths]

    if limit:
        image_paths = image_paths[:limit]
        labels = labels[:limit]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(1)

    return dataset

def save_visualized_images(images, labels, output_dir, prefix="image"):
    """Saves visualized images with labels to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # Normalize image to [0, 1] for saving
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)  # Ensure pixel values are between 0 and 1

        # Save image
        img_path = os.path.join(output_dir, f"{prefix}_{i}.png")
        plt.imsave(img_path, img)

def main():
    dataset_dir = "E:\deepfake_detector\dataset"
    
    # Load datasets with limit for training
    train_dataset = load_dataset(dataset_dir, 'Train', batch_size=32, limit=10000)
    valid_dataset = load_dataset(dataset_dir, 'Validation', batch_size=32, limit=10000)

    # Visualize a few images from the training dataset
    for images, labels in train_dataset.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        # Normalize images to [0, 1] for visualization
        images_np = (images_np + 1.0) / 2.0
        
        # Create titles for each image
        titles = [f"Label: {'Fake' if label == 1 else 'Real'}" for label in labels_np]
        
        # Debug print to verify labels
        for i, (img_title, img_label) in enumerate(zip(titles, labels_np)):
            print(f"Train Image {i}: {img_title}, Actual label: {'Fake' if img_label == 1 else 'Real'}")
        
        # Save and visualize images
        save_visualized_images(images_np, labels_np, output_dir='train_visualizations')
        visualize_images(images_np, titles)

    # Visualize a few images from the Validation dataset
    for images, labels in valid_dataset.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        # Normalize images to [0, 1] for visualization
        images_np = (images_np + 1.0) / 2.0
        
        # Create titles for each image
        titles = [f"Label: {'Fake' if label == 1 else 'Real'}" for label in labels_np]
        
        # Debug print to verify labels
        for i, (img_title, img_label) in enumerate(zip(titles, labels_np)):
            print(f"Valid Image {i}: {img_title}, Actual label: {'Fake' if img_label == 1 else 'Real'}")
        
        # Save and visualize images
        save_visualized_images(images_np, labels_np, output_dir='valid_visualizations')
        visualize_images(images_np, titles)

if __name__ == "__main__":
    main()
