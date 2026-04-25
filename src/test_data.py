from src.dataset import get_dataloaders
import torch
train_loader, test_loader, classes = get_dataloaders(
    train_dir="dataset/fer2013/train",
    test_dir="dataset/fer2013/test",
    batch_size=32
)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))