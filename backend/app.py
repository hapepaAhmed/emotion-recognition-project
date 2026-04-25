from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import cv2
from torchvision import transforms

from models.cnn.cnn_model import CNN
from models.MobileNet.mobilenet_model import MobileNetModel
from models.EfficientNet.efficientnet_model import EfficientNetModel


app = Flask(__name__)
CORS(app)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LABELS ----------------
EMOTIONS = ["Happy", "Neutral", "Sad"]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- GLOBAL VARIABLES ----------------
model = None
current_model_name = None


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

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    return jsonify({
        "emotion": EMOTIONS[pred.item()],
        "model": current_model_name
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)