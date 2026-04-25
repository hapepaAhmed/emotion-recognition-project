import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from models.cnn.cnn_model import CNN


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# Load data
train_loader, test_loader, classes = get_dataloaders(
    train_dir="dataset/fer2013/train",
    test_dir="dataset/fer2013/test",
    batch_size=32
)

# Model
model = CNN(num_classes=3).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler (important for better accuracy)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

epochs = 10


for epoch in range(epochs):
    # ===== TRAIN =====
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # ===== EVAL =====
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {running_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), "models/cnn/cnn_model.pth")