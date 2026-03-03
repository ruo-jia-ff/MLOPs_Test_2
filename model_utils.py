# model_utils.py

import os
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

def get_data_loaders(data_dir: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepares data loaders for training and testing.
    """
    grey_normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        grey_normalize
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        grey_normalize
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transforms)

    # Save label map
    with open("label_map.json", "w") as f:
        json.dump(train_dataset.classes, f)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, len(train_dataset.classes)

class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for grayscale image classification.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.dropout_fc = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        return self.fc2(x)


def train_model(model, train_loader, test_loader, device, epochs=10, lr=1e-3, step_size=4):
    """
    Trains and evaluates the model over multiple epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        print(f"[Epoch {epoch}/{epochs}] Avg Train Loss: {avg_train_loss:.4f}", end=' | ')

        # Evaluation
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(test_loader.dataset)
        val_acc = correct / total * 100
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")


def export_model_onnx(model, output_path="rps_model.onnx", input_size=(1, 1, 128, 128)):
    """
    Exports the trained model to ONNX format.
    """
    model.eval()
    model.cpu()
    dummy_input = torch.randn(*input_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11
    )
    print(f"Model exported to {output_path}")