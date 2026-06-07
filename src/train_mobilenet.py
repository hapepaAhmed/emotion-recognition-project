import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders

from models.MobileNet.mobilenet_model import MobileNetModel


import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == 'cuda':
    cudnn.benchmark = True

# Data
train_loader, val_loader, classes = get_dataloaders(
       train_dir="dataset/fer2013/train",
       test_dir="dataset/fer2013/test",
)


# Model
model = MobileNetModel().to(device)


# Loss (with Class Weights for imbalance)
class_weights = torch.tensor([1.0, 1.5, 2.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer (lower LR = better for pretrained models, add weight decay)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Scheduler (VERY IMPORTANT for MobileNet)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

epochs = 15
best_val_acc = 0.0

for epoch in range(epochs):

    # ================= TRAIN =================
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


    # ================= VALIDATION =================
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total


    # Step scheduler
    scheduler.step(val_acc)


    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {running_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/MobileNet/mobilenet_model.pth")
        print(f"--> Saved new best MobileNet with Val Acc: {best_val_acc:.2f}%")