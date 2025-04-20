import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Paramètres
data_dir = './dechets_classes'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Datasets
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'seg_train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'seg_test'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'seg_pred'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.classes)

# VGG16 modifié
def build_vgg16():
    model = models.vgg16(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(25088, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    return model.to(device)

# Fonction d'entraînement/évaluation
def train_and_evaluate(model, name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

    train_acc, val_acc = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_corrects = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_corrects += (outputs.argmax(1) == labels).sum().item()

        epoch_acc = running_corrects / len(train_loader.dataset)
        train_acc.append(epoch_acc)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())

        val_epoch_acc = accuracy_score(all_labels, all_preds)
        val_acc.append(val_epoch_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train acc: {epoch_acc:.4f} - Val acc: {val_epoch_acc:.4f}")

    # Résultats finaux
    kappa = cohen_kappa_score(all_labels, all_preds)
    print(f"\nClassification Report sur validation:\n{classification_report(all_labels, all_preds)}")
    print(f"Kappa final: {kappa:.4f}")

    # Courbe
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title(f'Courbe d\'apprentissage - {name}')
    plt.xlabel('Époques')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prédictions seg_pred
    pred_classes = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            pred_classes.extend(outputs.argmax(1).cpu().numpy())
    print("Exemples de prédictions seg_pred:", pred_classes[:5])

    # Sauvegarde
    torch.save(model.state_dict(), f"{name.replace(' ', '_').lower()}.pth")
    print(f"Modèle sauvegardé sous {name.replace(' ', '_').lower()}.pth")

# Lancer l'entraînement
model = build_vgg16()
train_and_evaluate(model, "VGG16 Transfer Learning")
