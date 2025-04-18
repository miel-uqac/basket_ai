import os
import urllib.request

# URLs des fichiers modèles
prototxt_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"
caffemodel_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

# Noms des fichiers locaux
prototxt_filename = "MobileNetSSD_deploy.prototxt"
caffemodel_filename = "MobileNetSSD_deploy.caffemodel"

# Fonction pour télécharger un fichier
def download_file(url, filename):
    if not os.path.isfile(filename):
        print(f"Téléchargement de {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} téléchargé avec succès.")
    else:
        print(f"{filename} existe déjà.")

# Télécharger les fichiers
download_file(prototxt_url, prototxt_filename)
download_file(caffemodel_url, caffemodel_filename)

# Vérifier les tailles des fichiers pour s'assurer qu'ils ne sont pas corrompus
prototxt_size = os.path.getsize(prototxt_filename)
caffemodel_size = os.path.getsize(caffemodel_filename)
print(f"Taille de {prototxt_filename}: {prototxt_size} octets")
print(f"Taille de {caffemodel_filename}: {caffemodel_size} octets")

if prototxt_size < 1000 or caffemodel_size < 100000:	
    print("Erreur : l'un des fichiers téléchargés est corrompu ou incomplet.")
else:
    print("Les fichiers semblent être corrects.")

