import torch
from torchvision import models, transforms
from PIL import Image
import os

# Paramètres
IMG_SIZE = 128
MODEL_PATH = "vgg16_transfer_learning.pth"  # nom du fichier sauvegardé
CLASSES = ['autre','bouteille','canettes','rien']  # À adapter selon ton dataset

# Définition des mêmes transformations qu'à l'entraînement
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Fonction pour construire le modèle (doit correspondre au modèle sauvegardé)
def build_vgg16(num_classes=4):
    model = models.vgg16(pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(25088, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(128, num_classes)
    )
    return model

def predict(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger l'image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Charger le modèle
    model = build_vgg16(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Prédiction
    with torch.no_grad():
        output = model(image_tensor)
        pred_class = output.argmax(1).item()
        print(f"Classe prédite : {CLASSES[pred_class]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage : python predict_image.py chemin/vers/image.jpg")
    else:
        predict(sys.argv[1])
