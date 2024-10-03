import os
import numpy as np
import tensorflow as tf
import cv2
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
try:
    model = tf.keras.models.load_model(
        'E:\\deepfake_detector\\model\\deepfake_video_model.h5')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# Constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Build feature extractor (InceptionV3)


def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

# Load video frames and prepare for prediction


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []

    if not cap.isOpened():
        logging.error("Error opening video file.")
        return []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, resize)
            frames.append(frame)

            # Limit the number of frames processed
            if max_frames > 0 and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    return np.array(frames)

# Prepare video for prediction


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i in range(len(frames)):
        video_length = frames[i].shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(
                frames[i][None, j, :])
        frame_mask[i, :length] = 1

    return frame_features, frame_mask

# Cleanup function to remove uploaded video files


def cleanup_video(video_path):
    if os.path.exists(video_path):
        os.remove(video_path)
        logging.info(f"Deleted temporary file: {video_path}")

# Prediction endpoint

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    try:
        if video.content_type not in ["video/mp4", "video/mov"]:
            raise HTTPException(status_code=400, detail="Invalid video format")

        # Save the uploaded video temporarily
        uploads_dir = os.path.join(os.getcwd(), "api", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        video_path = os.path.join(uploads_dir, video.filename)

        with open(video_path, "wb") as f:
            f.write(await video.read())
        
        logging.info(f"Uploaded video saved at: {video_path}")

        # Load and process video frames (limit max_frames if needed)
        frames = load_video(video_path)

        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="Could not read video")

        # Prepare video for prediction
        frame_features, frame_mask = prepare_single_video(frames)

        # Make prediction
        prediction = model.predict([frame_features, frame_mask])[0]
        result = 'FAKE' if prediction >= 0.51 else 'REAL'
        confidence = float(prediction)

        # Log prediction result
        logging.info(f"Prediction result: {result}, Confidence: {confidence}")

        # Schedule cleanup of the uploaded file after processing is complete
        background_tasks.add_task(cleanup_video, video_path)

        return JSONResponse(content={"Result": result, "Confidence": confidence})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during analysis.")