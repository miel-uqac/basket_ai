import numpy as np
import time
from picamera2 import Picamera2
import cv2

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
photo_taken_time = 0  # Variable pour garder la trace du moment où une photo a été prise
photo_delay = 0.05  # Temps d'attente en secondes avant de permettre un nouveau mode burst
timestamp = 0

# Paramètres du mode burst
burst_count = 5  # Nombre d'images à prendre en rafale
burst_delay = 0.001  # Délai entre chaque prise d'image en rafale (en secondes)

picam2.set_controls({"ExposureTime": 1500})
while True:
    # Capture de l'image actuelle de la caméra
    current_frame = picam2.capture_array("main")  # Capture en tant que tableau NumPy

    # Conversion en niveau de gris pour simplifier la détection de mouvement
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    if prev_frame is not None:
        # Calcul de la différence absolue entre l'image précédente et l'image actuelle
        frame_diff = cv2.absdiff(prev_frame, gray_frame)

        # Appliquer un seuillage pour mettre en évidence les régions avec une différence importante
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        # Comptage du nombre de pixels blancs (différence significative)
        non_zero_pixels = np.count_nonzero(thresh)

        # Si le nombre de pixels différents dépasse le seuil, il y a du mouvement
        if non_zero_pixels > motion_threshold:
            # Vérifier si un délai est passé depuis la dernière prise de photo
            current_time = time.time()
            if current_time - photo_taken_time > photo_delay:
                print("Mouvement détecté! Prise de photo en rafale.")

                # Capture en rafale de plusieurs images
                for i in range(burst_count):
                    timestamp += 1
                    photo_filename = f"mouvement_{timestamp}.jpg"
                    picam2.capture_file(f"motion_cap/{photo_filename}")
                    print(f"Photo {i + 1} enregistrée sous {photo_filename}")

                    # Attendre un peu avant de prendre la suivante
                    time.sleep(burst_delay)

                # Mettre à jour le temps de la dernière photo prise
                photo_taken_time = current_time
            else:
                print("Mouvement détecté mais photo déjà prise récemment.")

    # Sauvegarder la frame actuelle comme précédente pour la prochaine itération
    prev_frame = gray_frame

    # Attente pour éviter une boucle trop rapide
    time.sleep(0.1)
