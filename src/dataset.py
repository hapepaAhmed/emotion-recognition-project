import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(train_dir, test_dir, batch_size=32):
    """
    Returns train and test dataloaders
    """

    # Device check (optional but useful)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),  # FER is grayscale → convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Dataloaders
    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("Classes:", train_dataset.classes)
    print("Class to index:", train_dataset.class_to_idx)

    return train_loader, test_loader, train_dataset.classes