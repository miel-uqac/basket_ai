import os
import shutil
import random

# Chemins
source_dir = 'dechets'
target_dir = 'dechets_classes'
splits = ['seg_train', 'seg_test', 'seg_pred']
split_ratios = [0.7, 0.2, 0.1]  # train, test, pred

# Créer les dossiers cibles
classes = os.listdir(source_dir)
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

# Répartition des fichiers
for cls in classes:
    class_path = os.path.join(source_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(split_ratios[0] * n_total)
    n_test = int(split_ratios[1] * n_total)

    for i, img in enumerate(images):
        src_path = os.path.join(class_path, img)
        if i < n_train:
            split = 'seg_train'
        elif i < n_train + n_test:
            split = 'seg_test'
        else:
            split = 'seg_pred'
        dst_path = os.path.join(target_dir, split, cls, img)
        shutil.copyfile(src_path, dst_path)

print("Répartition terminée !")
