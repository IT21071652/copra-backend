from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the grading model
GRADING_MODEL_PATH = "models/grading/resnet50/copra_grading_identification_resnet50.h5"
grading_model = tf.keras.models.load_model(GRADING_MODEL_PATH)
grading_labels = ['Class1', 'Class2', 'Class3']  # Replace with actual class names

# Load the mold detection model
MOLD_MODEL_PATH = "models/mold/resnet50/copra_mold_identification_resnet50.h5"
mold_model = tf.keras.models.load_model(MOLD_MODEL_PATH)
mold_labels = ['Moldy', 'Healthy']  # Replace with actual class names

# Load the TFLite model
TFLITE_MODEL_PATH = "models/mold/resnet50/copra_mold_identification_resnet50_quant.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
