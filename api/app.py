import os
import numpy as np
import tensorflow as tf
import cv2
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__, template_folder="../web/templates", static_folder="../web/static")

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'api', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
try:
    model = tf.keras.models.load_model('model/deepfake_video_model_2.keras')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# Constants for video processing
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
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if video_file.content_type not in ["video/mp4", "video/mov"]:
            return jsonify({"error": "Invalid video format"}), 400

        # Save the uploaded video temporarily
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        
        logging.info(f"Uploaded video saved at: {video_path}")

        # Load and process video frames (limit max_frames if needed)
        frames = load_video(video_path)

        if len(frames) == 0:
            return jsonify({"error": "Could not read video"}), 400

        # Prepare video for prediction
        frame_features, frame_mask = prepare_single_video(frames)

        # Make prediction
        prediction = model.predict([frame_features, frame_mask])[0]
        result = 'FAKE' if prediction >= 0.51 else 'REAL'
        confidence = float(prediction)

        # Log prediction result
        logging.info(f"Prediction result: {result}, Confidence: {confidence}")

        # Cleanup the uploaded file after processing is complete
        cleanup_video(video_path)

        return jsonify({"Result": result, "Confidence": confidence})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during analysis."}), 500

# Route to render the main page with the upload form
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)