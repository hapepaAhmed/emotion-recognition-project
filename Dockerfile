FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || echo "Ignoring errors in requirements"

# Install other known requirements if not present
RUN pip install flask flask-cors torch torchvision opencv-python-headless numpy

COPY . .

EXPOSE 5000
ENV PYTHONPATH=/app

CMD ["python", "backend/app.py"]
