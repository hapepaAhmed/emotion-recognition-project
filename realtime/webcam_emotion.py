import cv2
import torch
import numpy as np
from torchvision import transforms

from models.cnn.cnn_model import CNN
from models.MobileNet.mobilenet_model import MobileNetModel
from models.EfficientNet.efficientnet_model import EfficientNetModel


# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LABELS ----------------
EMOTIONS = ["Happy", "Neutral", "Sad"]


# ---------------- TRANSFORM (OPTIMIZED FOR SPEED) ----------------
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


# ---------------- LOAD MODEL FUNCTION ----------------
def load_model(model_name):

    if model_name == "cnn":
        model = CNN(num_classes=3)
        model.load_state_dict(torch.load("models/cnn/cnn_model.pth", map_location=device))

    elif model_name == "mobilenet":
        model = MobileNetModel(num_classes=3)
        model.load_state_dict(torch.load("models/.MobileNet/mobilenet_model.pth", map_location=device))

    elif model_name == "efficientnet":
        model = EfficientNetModel(num_classes=3)
        model.load_state_dict(torch.load("models/EfficientNet/efficientnet_model.pth", map_location=device))

    else:
        raise ValueError("Invalid model name")

    model.to(device)
    model.eval()

    return model


# ---------------- CHOOSE MODEL HERE ----------------
CURRENT_MODEL_NAME = "cnn"   # change this later (or from frontend)
model = load_model(CURRENT_MODEL_NAME)


# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)

print("Press:")
print("1 -> CNN")
print("2 -> MobileNet")
print("3 -> EfficientNet")
print("q -> Quit")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    # ---------------- SWITCH MODEL LIVE ----------------
    if key == ord('1'):
        model = load_model("cnn")
        print("Switched to CNN")

    elif key == ord('2'):
        model = load_model("mobilenet")
        print("Switched to MobileNet")

    elif key == ord('3'):
        model = load_model("efficientnet")
        print("Switched to EfficientNet")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            face_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(face_tensor)
                _, pred = torch.max(outputs, 1)
                emotion = EMOTIONS[pred.item()]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        except Exception as e:
            print("Error:", e)

    cv2.imshow("Emotion Detection (PyTorch)", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()