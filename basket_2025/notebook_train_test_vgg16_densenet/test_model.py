import numpy as np
import time
from picamera2 import Picamera2
import cv2
import os
import torch
import torch.nn as nn
from torchvision import models
import torchvision.models as models
from PIL import Image
from torchvision import transforms


# Initialiser la caméra
picam2 = Picamera2()

# Configuration de la caméra pour obtenir une image en RGB
video_config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(video_config)
picam2.start()


prev_frame = None
motion_threshold = 6000  # Valeur seuil pour détecter le mouvement
ltime = time.time()
photo_taken_time = 0  
photo_delay = 0.05  

# Paramètres du mode burst
burst_count = 5  # Nombre d'images à prendre en rafale
burst_delay = 0.001  # Délai entre chaque prise d'image en rafale (en secondes)

picam2.set_controls({"ExposureTime": 1500})

# Transformation pour DenseNet121
densenet121_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model_densenet121 = models.densenet121(pretrained=True)
model_densenet121.classifier = nn.Linear(model_densenet121.classifier.in_features, 4)
model_densenet121.load_state_dict(torch.load("densenet_ourdata.pth"))
model_densenet121.eval()

# Transformation pour VGG16
vgg16_transform = transforms.Compose([
    transforms.Resize((128, 128)), #entraine en 128 par 128 et non en 224 par 224 comme densenet, a modifie si on change les parametres d'entrainement
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model_vgg = models.vgg16(pretrained=True)
model_vgg.classifier[6] = nn.Linear(model_vgg.classifier[6].in_features, 4)
model_vgg.load_state_dict(torch.load("vgg16_ourdata.pth"))
model_vgg.eval()

class_map = {0: "autres", 1: "bouteilles", 2: "canettes", 3: "rien"}

nom_model= input("Selectionner votre modele : taper 'vgg16' ou 'densenet' \n")

if nom_model=="densenet":
    model=model_densenet121
    transform=densenet121_transform
if nom_model=="vgg16":
    model=model_vgg
    transform=vgg16_transform
else:
    raise ValueError("Modele non reconnu, taper 'vgg16' ou 'densenet'\n")
while True :
       
    current_frame = picam2.capture_array("main")
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        non_zero_pixels = np.count_nonzero(thresh)

        if non_zero_pixels > motion_threshold:
            current_time = time.time()
            if current_time - photo_taken_time > photo_delay:
                print("Mouvement détecté! Prise de photo en rafale.")
                for i in range(burst_count):
                    photo_filename = f"mouvement_{i}.jpg"
                    picam2.capture_file(f"motion_cap/{photo_filename}")
                    print(f"Photo {i + 1} enregistrée sous {photo_filename}")
                    time.sleep(burst_delay)
                photo_taken_time = current_time
                preds = [0,0,0,0]
                for file in os.listdir("motion_cap"):
                    # Application des transformations
                    file_path = os.path.join("motion_cap",file) 
                    img=Image.open(file_path)
                    img_transfo = transform(img)

                    # Ajout d'une dimension pour le batch
                    img_transfo = img_transfo.unsqueeze(0)

                    with torch.no_grad():
                        output= model(img_transfo)
                        _, pred = torch.max(output, 1)
                        print(pred)
                    preds[pred.item()] += 1
                
                pred_max = 0
                indice=0
                for i in range(len(preds)) :
                    if preds[i] > pred_max:
                        indice = i
                        pred_max = preds[i]

                # Affichage de la prédiction
                print(f"Prédiction : {class_map[indice]}")
                time.sleep(20)
                print("nouvelle prediction prete")
            else:
                print("Mouvement détecté mais photo déjà prise récemment.")

    prev_frame = gray_frame

picam2.close()