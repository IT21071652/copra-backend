from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from flask_cors import CORS
import gdown

app = Flask(__name__)
CORS(app)

# Model paths
GRADING_MODEL_PATH = "models/grading/resnet50/copra_grading_identification_resnet50.h5"
MOLD_MODEL_PATH = "models/mold/resnet50/copra_mold_identification_resnet50.h5"
TFLITE_MODEL_PATH = "models/mold/resnet50/copra_mold_identification_resnet50_quant.tflite"

# Model labels
grading_labels = ['Grade A', 'Grade B', 'Grade C', 'Grade D']
mold_labels = ['Moldy', 'Not Moldy']

def download_model(file_id, output_path):
    """Download model from Google Drive if not exists"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(output_path):
            url = f'https://drive.google.com/uc?id={file_id}'
            return gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

# Initialize models
grading_model = None
mold_model = None
interpreter = None
input_details = None
output_details = None

def initialize_models():
    """Initialize models based on environment"""
    global grading_model, mold_model, interpreter, input_details, output_details
    
    if os.environ.get('FLASK_ENV') == 'production':
        print("Downloading models from Google Drive...")
        models_downloaded = all([
            download_model(os.environ.get('GRADING_MODEL_ID'), GRADING_MODEL_PATH),
            download_model(os.environ.get('MOLD_MODEL_ID'), MOLD_MODEL_PATH),
            download_model(os.environ.get('TFLITE_MODEL_ID'), TFLITE_MODEL_PATH)
        ])
        if not models_downloaded:
            print("Failed to download models")
            return False
    
    try:
        print("Loading models...")
        grading_model = tf.keras.models.load_model(GRADING_MODEL_PATH)
        mold_model = tf.keras.models.load_model(MOLD_MODEL_PATH)
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Initialize models on startup
if not initialize_models():
    print("Warning: Failed to initialize models")

# Image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict_grading', methods=['POST'])
def predict_grading():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_path = "temp.jpg"
    file.save(file_path)
    
    img_array = preprocess_image(file_path)
    predictions = grading_model.predict(img_array)
    predicted_class = grading_labels[np.argmax(predictions)]
    
    os.remove(file_path)

    return jsonify({"predicted_class": predicted_class})

@app.route('/predict_mold', methods=['POST'])
def predict_mold():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    file_path = "temp.jpg"
    file.save(file_path)
    
    img_array = preprocess_image(file_path)
    
    # Predict with Keras model
    predictions = mold_model.predict(img_array)
    predicted_class = mold_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    
    os.remove(file_path)

    return jsonify({'class': predicted_class, 'confidence': confidence})

@app.route('/predict_tflite', methods=['POST'])
def predict_tflite():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_path = "temp.jpg"
    file.save(file_path)
    
    img_array = preprocess_image(file_path).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = mold_labels[np.argmax(output_data)]
    confidence = float(np.max(output_data))

    os.remove(file_path)

    return jsonify({'class': predicted_class, 'confidence': confidence})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
