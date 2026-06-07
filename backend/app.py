from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import cv2
import os
from torchvision import transforms

from models.cnn.cnn_model import CNN
from models.MobileNet.mobilenet_model import MobileNetModel
from models.EfficientNet.efficientnet_model import EfficientNetModel


app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/stream.html')
def stream_page():
    return app.send_static_file('stream.html')

@app.route('/style.css')
def style():
    return app.send_static_file('style.css')

@app.route('/script.js')
def script():
    return app.send_static_file('script.js')

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LABELS ----------------
EMOTIONS = ["Happy", "Neutral", "Sad"]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ---------------- GLOBAL VARIABLES ----------------
model = None
current_model_name = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ---------------- LOAD MODEL ----------------
def load_model(name):
    global model, current_model_name

    current_model_name = name

    if name == "cnn":
        model = CNN(num_classes=3)
        model.load_state_dict(torch.load("models/cnn/cnn_model.pth", map_location=device))

    elif name == "mobilenet":
        model = MobileNetModel(num_classes=3)
        model.load_state_dict(torch.load("models/MobileNet/mobilenet_model.pth", map_location=device))

    elif name == "efficientnet":
        model = EfficientNetModel(num_classes=3)
        model.load_state_dict(torch.load("models/EfficientNet/efficientnet_model.pth", map_location=device))

    else:
        return False

    model.to(device)
    model.eval()
    return True


# ---------------- SELECT MODEL (FROM FRONTEND) ----------------
@app.route("/select_model", methods=["POST"])
def select_model():
    data = request.json
    model_name = data.get("model")

    success = load_model(model_name)

    if not success:
        return jsonify({"error": "Invalid model"}), 400

    return jsonify({
        "status": "model loaded",
        "model": model_name
    })


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    global model

    if model is None:
        return jsonify({"error": "No model selected"}), 400

    file = request.files["image"]

    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Convert BGR to RGB so ToPILImage reads it correctly
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return jsonify({"error": "No face detected"})
        
    # Crop the first face found
    x, y, w, h = faces[0]
    face_img = img_rgb[y:y+h, x:x+w]

    # Transform the cropped face
    face_tensor = transform(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(face_tensor)
        _, pred = torch.max(outputs, 1)

    return jsonify({
        "emotion": EMOTIONS[pred.item()],
        "model": current_model_name
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)