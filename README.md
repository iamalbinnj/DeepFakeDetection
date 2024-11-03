# DEEP FAKE IMAGE DETECTION

## Description

This project focuses on detecting deepfakes in videos using a CNN-RNN architecture. A pre-trained InceptionV3 model extracts features from video frames, which are then fed into a GRU layer to capture temporal information. The model is trained on a labeled dataset and evaluated on test videos. The code includes functions for data preparation, model training, and prediction.

## Technologies Used

- **TensorFlow**: An open-source library for machine learning and deep learning models.
- **Keras**: A high-level neural networks API, running on top of TensorFlow.
- **OpenCV**: An open-source computer vision and machine learning software library.
- **NumPy**: A fundamental package for scientific computing in Python.
- **Flask**: A lightweight WSGI web application framework used to build the REST API for the project.

## Dataset

The dataset used in this project consists of deepfake and real video and is sourced from Kaggle. You can download the dataset from the following link:

[Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)

### Guide to Download the Dataset

1. Visit the [Kaggle Dataset Page](https://www.kaggle.com/competitions/deepfake-detection-challenge/data).
2. Log in to your Kaggle account.
3. Click on the "Download" button to download the dataset.
4. Extract the downloaded `.zip` file and place it in the `dataset` directory within the project structure.

## File Structure

The project is organized as follows:

```
deepfake_detector/
│
├── api/                        
│   ├── uploads/                
│   └── app.py                 
│
├── dataset/                    
│   ├── static/                 
│   ├── test_videos/            
│   └── train_sample_videos/    
│
├── model/                      
│
├── src/                        
│   ├── notebook/              
│       └── main.ipynb          
│
├── web/                        
│   ├── static/                
│   │   ├── css/                
│   │   │   └── style.css       
│   │   ├── js/                 
│   │       └── script.js       
│   └── templates/              
│       └── index.html          
│
├── .gitignore                  
├── README.md                   
└── requirements.txt   

```

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed on your system:

- **Python 3.7 or above**
- **Git** (for cloning the repository)

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

4. **Create Environment File**:
   After installing the dependencies, create the `.env` file by running:
   ```bash
   cp .env.example .env && rm .env.example
   ```

5. **Deactivate the Virtual Environment**

   After working on the project, you can deactivate the virtual environment with the following command:
   ```sh
   deactivate
   ```

### Running the Flask App

After installing the dependencies run the ```app.py``` file to start the Flask API:
```sh
   python api/app.py
```

## Challenges and Future Work

### Challenges
- **Data Quality**: Deepfake images can vary significantly in quality, making it challenging for the model to generalize.
- **Overfitting**: With a limited amount of labeled data, the model may overfit to the training data, reducing its performance on unseen data.
- **Model Complexity**: Balancing the complexity of the model with computational efficiency was a significant challenge.

### Future Work
- **Dataset Expansion**: Incorporate a larger and more diverse dataset to improve the model's robustness.
- **Transfer Learning**: Experiment with transfer learning using pre-trained models on large image datasets to improve accuracy.

## Conclusion
The "Deep Fake Detection" project demonstrates the effectiveness of deep learning models in distinguishing between real and fake videos. The project provides a foundation for further research and development in the field of deepfake detection.
