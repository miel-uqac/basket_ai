classeur.py ne sert que à creer dechets_classes

lancer train_vgg16_pytorch.py directement depuis votre IDE sans commande particulière une fois que vous aurez mis les paramètre d'entrainement que vous souhaitez.

Pour lancer test_vgg16.py, executer : python test_vgg16.py chemin_image/image.jpg

tout les codes doivent être lancer au niveau du dossier vgg16, pas de venv n'est nécessaire

Le modèle étant trop lourd, il faut lancer l'entrainement une première fois afin de créer le modèle, puis deplacer une copie de celui-ci dans le dossier notebook afin de pouvoir faire des tests.