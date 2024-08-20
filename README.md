# DEEP FAKE IMAGE DETECTION

## Description

This project aims to detect deepfake images using a convolutional neural network (CNN) model. Deepfakes are synthetic media in which a person in an existing image or video is replaced with someone else's likeness. The proliferation of deepfakes poses significant ethical and security challenges, making the ability to detect such media crucial. This project leverages machine learning techniques to distinguish between real and fake images with a high degree of accuracy.

## Technologies Used

- **TensorFlow**: An open-source library for machine learning and deep learning models.
- **Keras**: A high-level neural networks API, running on top of TensorFlow.
- **OpenCV**: An open-source computer vision and machine learning software library.
- **NumPy**: A fundamental package for scientific computing in Python.

## Dataset

The dataset used in this project consists of deepfake and real images and is sourced from Kaggle. You can download the dataset from the following link:

[Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

### Guide to Download the Dataset

1. Visit the [Kaggle Dataset Page](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images).
2. Log in to your Kaggle account.
3. Click on the "Download" button to download the dataset.
4. Extract the downloaded `.zip` file and place it in the `dataset` directory within the project structure.

## File Structure

The project is organized as follows:

```
deepfake_detector/
│
├── models/
│ └── deepfake_detector_model.h5
|
├── notebook/
│ ├── create_model.ipynb
│ ├── preprocess.ipynb
│ ├── train_model.ipynb
│ └── visualize.ipynb
|
├── src/
│ ├── create_model.py
│ ├── predict_image.py
│ ├── preprocess.py
│ ├── test_model.py
│ ├── train_model.py
| ├── validate_model.py
│ └── visualize.py
|
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed on your system:

- **Python 3.7 or above**
- **Git** (for cloning the repository)

### Creating and Using a Virtual Environment

To isolate your project’s dependencies, it’s recommended to use a virtual environment.

1. **Create a Virtual Environment**

   Run the following command to create a virtual environment named `env`:

   ```sh
   python -m venv deepfake_env
   ```

2. **Activate the Virtual Environment**
   * Windows:
      ```sh
      .\deepfake_env\Scripts\activate
       ```
   * macOS/Linux:
      ```sh
      source deepfake_env/bin/activate
      ``` 

3. **Install Dependencies**
   
   Once the virtual environment is activated, install the project dependencies using the `requirements.txt` file:

   ```sh
   pip install -r requirements.txt
   ```

4. **Deactivate the Virtual Environment**

   After working on the project, you can deactivate the virtual environment with the following command:
   ```sh
   deactivate
   ```

### Installation

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**

   First, clone the repository using Git:

   ```sh
   https://github.com/iamalbinnj/DeepFakeDetection.git
   ```

2. **Go to the project folder**

   ```sh
   cd DeepFakeDetection
   ```

3. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

## Challenges and Future Work

### Challenges
- **Data Quality**: Deepfake images can vary significantly in quality, making it challenging for the model to generalize.
- **Overfitting**: With a limited amount of labeled data, the model may overfit to the training data, reducing its performance on unseen data.
- **Model Complexity**: Balancing the complexity of the model with computational efficiency was a significant challenge.

### Future Work
- **Dataset Expansion**: Incorporate a larger and more diverse dataset to improve the model's robustness.
- **Real-time Detection**: Extend the project to support real-time deepfake detection in video streams.
- **Transfer Learning**: Experiment with transfer learning using pre-trained models on large image datasets to improve accuracy.

## Conclusion
The "Deep Fake Image Detection" project demonstrates the effectiveness of deep learning models in distinguishing between real and fake images. Although the model achieves good accuracy, there are still challenges to address, particularly in terms of generalization and real-time application. The project provides a foundation for further research and development in the field of deepfake detection.
