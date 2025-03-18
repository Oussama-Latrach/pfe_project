import os
import laspy
import numpy as np


def load_multiple_laz_files(directory):
    all_points = []
    all_labels = []

    for filename in os.listdir(directory):
        if filename.endswith('.laz'):
            file_path = os.path.join(directory, filename)
            with laspy.open(file_path) as file:
                las = file.read()

                # Extraire les coordonnées des points
                points = np.vstack((las.x, las.y, las.z)).transpose()
                all_points.append(points)

                # Extraire les labels de classification
                labels = las.classification
                all_labels.append(labels)

    # Combiner tous les points et labels
    all_points = np.vstack(all_points)
    all_labels = np.concatenate(all_labels)

    return all_points, all_labels


def normalize_points(points):
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    normalized_points = (points - min_vals) / (max_vals - min_vals)
    return normalized_points


def map_labels(labels):
    # Mapping des labels
    label_mapping = {
        1: 'unclassified',  # Remplacez par l'index correspondant à 'unclassified'
        2: 'ground',
        3: 'low vegetation',
        4: 'medium vegetation',
        5: 'high vegetation',
        6: 'building',
        9: 'water',
        64: 'unclassified',  # Fusionner ces labels
        65: 'unclassified',
        202: 'unclassified',
        # Ajoutez d'autres mappings si nécessaire
    }

    # Remplacer les labels
    new_labels = []
    for label in labels:
        if label in label_mapping:
            new_labels.append(label_mapping[label])
        else:
            new_labels.append('unclassified')  # Pour tous les autres labels

    return np.array(new_labels)


# Chemin vers le répertoire contenant les fichiers .laz
directory = 'E:/cours_geomatique_3eme_annee/PFE/pratique/essaie1/data'  # Remplacez par le chemin de votre répertoire

# Charger les points et les labels
points, labels = load_multiple_laz_files(directory)

# Normaliser les points
normalized_points = normalize_points(points)

# Mapper les labels
new_labels = map_labels(labels)

# Sauvegarder les données
np.save('normalized_points.npy', normalized_points)
np.save('new_labels.npy', new_labels)

print("Données sauvegardées :")
print("normalized_points.npy")
print("new_labels.npy")