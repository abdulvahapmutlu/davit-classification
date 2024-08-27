import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset path
dataset_path = "C:/Users/offic/OneDrive/Masaüstü/datasets/caltech-101"

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for DaViT
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# Load DaViT model from Timm
model = timm.create_model('davit_tiny', pretrained=True)
num_features = model.head.in_features
model.head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # Pool to 1x1 spatial size
    nn.Flatten(),             # Flatten the tensor
    nn.Linear(num_features, len(dataset.classes))  # Adjust for Caltech-101 classes
)
model.to(device)

# Loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)
  
#Training the model.
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train = 0

    # Progress bar for the current epoch
    epoch_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

    for images, labels in epoch_bar:
        images, labels = images.to(device), labels.to(device)
        total_train += labels.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += calculate_accuracy(outputs, labels) * labels.size(0)

        # Update progress bar with current metrics
        epoch_bar.set_postfix({
            'loss': running_loss/total_train,
            'accuracy': running_corrects/total_train
        })

    epoch_loss = running_loss / total_train
    epoch_acc = running_corrects / total_train

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            total_val += labels.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            val_running_corrects += calculate_accuracy(outputs, labels) * labels.size(0)

    val_loss = val_running_loss / total_val
    val_acc = val_running_corrects / total_val

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
    scheduler.step()
