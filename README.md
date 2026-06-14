<div align="center">

# 🧠 Emotion Recognition AI

**Real-time facial emotion detection powered by three neural network architectures**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)

</div>

---

## 📌 Overview

This project is a **real-time facial emotion recognition system** that lets you capture a photo from your webcam and instantly receive an emotion prediction. It supports three neural network architectures — a Custom CNN, MobileNet V2, and EfficientNet B0 — each trained on the [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset to classify three emotions:

| Label | Emotion |
|-------|---------|
| 😄 | Happy |
| 😐 | Neutral |
| 😢 | Sad |

---

## ✨ Features

- 🎥 **Live webcam capture** with a clean, modern dark-mode UI
- 📸 **Capture-then-predict** workflow — snapshot is frozen and analyzed on demand
- 🧠 **3 model choices** — switch between CNN, MobileNet V2, or EfficientNet B0
- 🔍 **Face detection** using OpenCV Haar Cascade before inference
- 🐳 **Docker-ready** for instant deployment
- 🔄 **CI/CD pipeline** via GitHub Actions

---

## 🗂 Project Structure

```
emotion-recognition-project/
│
├── backend/
│   └── app.py               # Flask API server (model loading, face detection, prediction)
│
├── frontend/
│   ├── index.html            # Model selection page
│   ├── stream.html           # Webcam capture & prediction page
│   ├── style.css             # Dark-mode glassmorphism UI
│   └── script.js             # Capture → predict → restart logic
│
├── models/
│   ├── cnn/
│   │   ├── cnn_model.py      # 4-Block Custom CNN architecture
│   │   └── cnn_model.pth     # Trained weights
│   ├── MobileNet/
│   │   ├── mobilenet_model.py
│   │   └── mobilenet_model.pth
│   └── EfficientNet/
│       ├── efficientnet_model.py
│       └── efficientnet_model.pth
│
├── src/
│   ├── dataset.py            # DataLoader with augmentation
│   ├── train_cnn.py          # CNN training script
│   ├── train_mobilenet.py    # MobileNet training script
│   └── train_efficientnet.py # EfficientNet training script
│
├── .github/workflows/
│   └── ci-cd.yml             # GitHub Actions CI/CD pipeline
│
├── Dockerfile                # Container definition
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended for training)
- Webcam

### 1. Clone the Repository

```bash
git clone https://github.com/hapepaAhmed/emotion-recognition-project.git
cd emotion-recognition-project
```

### 2. Create & Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the **FER2013** dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it as:

```
dataset/
└── fer2013/
    ├── train/
    │   ├── Happy/
    │   ├── Neutral/
    │   └── Sad/
    └── test/
        ├── Happy/
        ├── Neutral/
        └── Sad/
```

### 5. Train the Models (Optional — skip if .pth files exist)

```powershell
# Custom CNN (20 epochs)
python -m src.train_cnn

# MobileNet V2 (15 epochs)
python -m src.train_mobilenet

# EfficientNet B0 (15 epochs)
python -m src.train_efficientnet
```

> Each script automatically saves only the **best** model (by validation accuracy) to its respective `models/` folder.

### 6. Run the Backend

```powershell
python -m backend.app
```

### 7. Open the App

Navigate to **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t emotion-recognition-app .

# Run the container
docker run -p 5000:5000 emotion-recognition-app
```

Then open **[http://localhost:5000](http://localhost:5000)**.

---

## 🔄 CI/CD Pipeline

Every push to `main` automatically triggers a **GitHub Actions** workflow that:
1. Sets up Python 3.9
2. Installs all dependencies
3. Builds the Docker image to verify nothing is broken

See [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml).

---

## 🧠 Model Architectures

### Custom CNN
A deep 4-block convolutional network trained from scratch.

```
Input (96×96×3)
  → Conv2d(32) → BN → ReLU → MaxPool
  → Conv2d(64) → BN → ReLU → MaxPool
  → Conv2d(128) → BN → ReLU → MaxPool
  → Conv2d(256) → BN → ReLU → MaxPool
  → Flatten → FC(512) → Dropout(0.5) → FC(3)
```

### MobileNet V2
Pretrained on ImageNet, fine-tuned by unfreezing the last 6 feature blocks. Classifier replaced with a custom dropout + linear head.

### EfficientNet B0
Pretrained on ImageNet, fine-tuned by unfreezing the last 4 feature blocks. Classifier replaced with a custom dropout + linear head.

---

## ⚙️ Training Strategy

All three models use the following advanced training setup:

| Technique | Detail |
|-----------|--------|
| **Data Augmentation** | RandomHorizontalFlip, RandomRotation(15°), RandomAffine, ColorJitter, RandomErasing |
| **Class Weights** | `[1.0, 1.5, 2.0]` to address class imbalance |
| **Optimizer** | Adam with `weight_decay=1e-4` |
| **LR Scheduler** | `ReduceLROnPlateau` — lowers LR only when accuracy plateaus |
| **Best Model Saving** | `.pth` saved only when a new validation accuracy high-score is hit |
| **GPU Optimization** | `cudnn.benchmark = True`, `pin_memory=True`, `num_workers=4`, `batch_size=64` |

---

## 🖥️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/select_model` | Load a model (`cnn`, `mobilenet`, `efficientnet`) |
| `POST` | `/predict` | Send an image, receive emotion prediction |
| `GET`  | `/` | Serve the frontend |

### Example `/predict` response

```json
{
  "emotion": "Happy",
  "model": "efficientnet"
}
```

---

## 📋 Requirements

```
torch
torchvision
torchaudio
numpy
pandas
matplotlib
scikit-learn
opencv-python
flask
flask-cors
pillow
tqdm
```

---

## 👩‍💻 Author

**Habiba Ahmed**
GitHub: [@hapepaAhmed](https://github.com/hapepaAhmed)

---

<div align="center">
Made with ❤️ and PyTorch
</div>
