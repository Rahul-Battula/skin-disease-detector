import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import time

from utils.dataset import HAM10000Dataset
from models.model import create_model

# ---- Config ----
DATA_DIR = '../data'
IMG_DIR1 = f'{DATA_DIR}/raw/HAM10000_images_part_1'
IMG_DIR2 = f'{DATA_DIR}/raw/HAM10000_images_part_2'
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(f"Using device: {DEVICE}")

    # ---- Load data ----
    train_df = pd.read_csv(f'{DATA_DIR}/processed/train.csv')
    val_df = pd.read_csv(f'{DATA_DIR}/processed/val.csv')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HAM10000Dataset(train_df, IMG_DIR1, IMG_DIR2, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, IMG_DIR1, IMG_DIR2, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ---- Class weights ----
    class_names = train_dataset.classes
    train_labels = train_df['dx'].map(train_dataset.class_to_idx).values
    class_weights = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # ---- Model, loss, optimizer ----
    model = create_model(num_classes=len(class_names), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    # ---- Training loop ----
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), '../models/best_model.pth')
            print(f"  -> Saved new best model (val_acc: {val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()