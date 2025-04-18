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
from sklearn.metrics import accuracy_score
import torch.optim as optim

# Initialiser la caméra
picam2 = Picamera2()

# Configuration de la caméra pour obtenir une image en RGB
video_config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(video_config)
picam2.start()

# Variables pour le calcul du mouvement
prev_frame = None
motion_threshold = 6000  # Valeur seuil pour détecter le mouvement
ltime = time.time()
photo_taken_time = 0  
photo_delay = 0.05  # Temps d'attente en secondes avant de permettre un nouveau mode burst
timestamp = 0

# Paramètres du mode burst
burst_count = 5  # Nombre d'images à prendre en rafale
burst_delay = 0.001  # Délai entre chaque prise d'image en rafale (en secondes)

picam2.set_controls({"ExposureTime": 1500})

def Prendre_photo():
    while True and timestamp<10:
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
                        timestamp += 1
                        photo_filename = f"mouvement_{timestamp}.jpg"
                        picam2.capture_file(f"motion_cap/{photo_filename}")
                        print(f"Photo {i + 1} enregistrée sous {photo_filename}")
                        time.sleep(burst_delay)
                    photo_taken_time = current_time
                else:
                    print("Mouvement détecté mais photo déjà prise récemment.")
        prev_frame = gray_frame
        time.sleep(0.1)
        
    picam2.close()

while True :
    nom_model= input("Selectionner votre modele : taper 'vgg16' ou 'densenet'")

    if nom_model=="vgg16":
        Prendre_photo()
        for file in os.listdir("motion_cap"):
            # Application des transformations
            file_path = os.path.join("motion_cap",file) 
            img=Image.open(file_path)
            img_vgg16 = vgg16_transform(img)

            # Ajout d'une dimension pour le batch
            img_vgg16 = img_vgg16.unsqueeze(0)

            with torch.no_grad():
                output_vgg16 = model_vgg16(img_vgg16)
                _, pred_vgg16 = torch.max(output_vgg16, 1)


            # Affichage des prédictions
            print(f"Prédiction VGG16: {class_map[pred_vgg16.item()]}")
            plt.imshow(img)
            plt.axis("off")
            plt.show()

    elif nom_model=="densenet" :
        Prendre_photo()
        for file in os.listdir("motion_cap"):
            # Application des transformations
            file_path = os.path.join("motion_cap",file) 
            img=Image.open(file_path)
            img_densenet121 = densenet121_transform(img)

            # Ajout d'une dimension pour le batch
            img_densenet121 = img_densenet121.unsqueeze(0)

            with torch.no_grad():
                output_densenet121 = model_densenet121(img_densenet121)
                _, pred_densenet121 = torch.max(output_densenet121, 1)

            # Affichage des prédictions
            print(f"Prédiction DenseNet121: {class_map[pred_densenet121.item()]}")
            plt.imshow(img)
            plt.axis("off")
            plt.show()

