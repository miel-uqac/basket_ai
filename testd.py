import cv2
import numpy as np
from picamera2 import Picamera2
import os

# Initialisation de Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (800, 600)}))
picam2.set_controls({
    "ExposureTime": 5000,  # Temps d'exposition rapide (5 ms)
    "AnalogueGain": 4.0,   # ISO élevé pour compenser
    "FrameRate": 50        # Augmentation de la cadence
})
picam2.start()

# Initialisation pour la détection de mouvement
previous_frame = None
photo_counter = 0
photo_directory = "photos"
os.makedirs(photo_directory, exist_ok=True)  # Créer un répertoire pour les photos

print("Démarrage du flux vidéo...")

while photo_counter < 1000:
    frame = picam2.capture_array()

    # Conversion en niveaux de gris et réduction du flou gaussien
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if previous_frame is None:
        previous_frame = gray
        continue

    # Calcul de la différence entre les cadres
    frame_delta = cv2.absdiff(previous_frame, gray)
    thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]  # Seuil ajusté
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Taille minimale ajustée
            continue

        # Calcul du cadre englobant (bounding box)
        (x, y, w, h) = cv2.boundingRect(contour)

        # Définir une marge autour du cadre
        margin = 150 #20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(w + 2 * margin, frame.shape[1] - x)
        h = min(h + 2 * margin, frame.shape[0] - y)

        # Dessiner un rectangle autour du mouvement
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extraire la région d'intérêt (ROI)
        roi = frame[y:y + h, x:x + w]

        # Sauvegarder la région d'intérêt
        if roi.size > 0:
            photo_path = os.path.join(photo_directory, f"photo_{photo_counter}.jpg")
            cv2.imwrite(photo_path, roi)
            print(f"Photo {photo_counter + 1} enregistrée : {photo_path}")
            photo_counter += 1

        if photo_counter >= 1000:
            break

    # Mettre à jour le cadre précédent
    previous_frame = gray

    # Afficher le flux vidéo
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer les ressources
cv2.destroyAllWindows()
picam2.stop()
