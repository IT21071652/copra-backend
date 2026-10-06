# Copra Backend API

A Flask-based backend for automated copra quality assessment using deep learning. This project classifies copra by grade and detects mold contamination from uploaded images, helping streamline agricultural quality checks with minimal manual effort.

## Overview

The Copra Backend API is designed to support image-based inspection for coconut/copra products. It uses TensorFlow and TensorFlow Lite models to process images and provide rapid predictions for:

- Copra grading (Grade A, B, C, D)
- Mold detection (Moldy / Not Moldy)
- Lightweight inference using a quantized TFLite model

The service is exposed through REST API endpoints and is suitable for integration with mobile apps, web apps, or inspection workflows.

## Key Features

- Image upload handling with Flask
- Deep learning-based copra grading prediction
- Mold detection using trained CNN models
- TensorFlow Lite inference for optimized performance
- Automatic model loading and optional remote model download
- CORS support for frontend integration
- Simple API documentation route at `/`

## Project Structure

```text
copra-backend/
├── backend.py
├── models/
│   ├── grading/
│   └── mold/
├── requirements.txt
├── api.http
├── .gitignore
└── README.md
```

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Flask-CORS
- gdown
- pyngrok

## Model Setup

The app expects trained model files inside the `models` directory:

- `models/grading/resnet50/copra_grading_identification_resnet50.h5`
- `models/mold/resnet50/copra_mold_identification_resnet50.h5`
- `models/mold/resnet50/copra_mold_identification_resnet50_quant.tflite`

If running in production, the app can download the required model files from Google Drive using environment variables:

```bash
export FLASK_ENV=production
export GRADING_MODEL_ID="<your_grading_model_id>"
export MOLD_MODEL_ID="<your_mold_model_id>"
export TFLITE_MODEL_ID="<your_tflite_model_id>"
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/IT21071652/copra-backend.git
cd copra-backend
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the backend:

```bash
python backend.py
```

## API Endpoints

### GET `/`
Returns API metadata and endpoint documentation.

### POST `/predict_grading`
Uploads an image and returns the predicted copra grade.

Request:
- Form-data field: `file`

Response example:

```json
{
  "predicted_class": "Grade A"
}
```

### POST `/predict_mold`
Uploads an image and returns mold detection class and confidence.

Request:
- Form-data field: `file`

Response example:

```json
{
  "class": "Moldy",
  "confidence": 0.981234
}
```

### POST `/predict_tflite`
Uploads an image and runs inference using the TensorFlow Lite model.

Request:
- Form-data field: `file`

Response example:

```json
{
  "class": "Not Moldy",
  "confidence": 0.95432
}
```

## Example Request with curl

```bash
curl -X POST http://127.0.0.1:5000/predict_grading \
  -F "file=@sample.jpg"
```

## Function Highlights

### `download_model(file_id, output_path)`
Downloads a pre-trained model from Google Drive when the file is missing. This supports remote deployment and reduces the need to ship large model files with the repository.

### `initialize_models()`
Loads the Keras classification models and TensorFlow Lite interpreter during startup. It ensures the app is ready to handle inference requests as soon as the backend runs.

### `preprocess_image(img_path)`
Resizes and normalizes uploaded images so they match the trained model input dimensions and preprocessing pipeline.

### `predict_grading()`
Processes a user-uploaded copra image and returns the predicted quality grade.

### `predict_mold()`
Runs the full Keras CNN mold-detection model and returns the class label and confidence score.

### `predict_tflite()`
Uses the compiled TensorFlow Lite model for lightweight inference with comparable output predictions.

## Use Cases

- Quality inspection in copra processing plants
- Automated agricultural grading support
- Mobile or web-based visual inspection tools
- AI-assisted quality control in export operations

## License

This project is intended for research and internal use unless otherwise specified by the project owner.

## Contact

For collaboration or project updates, connect with the repository maintainer or use the project’s available communication channels.
