import numpy as np

def sample_data(file_path, sample_fraction=0.1):
    # Charger les données
    data = np.load(file_path)

    # Calculer le nombre d'échantillons à prendre
    num_samples = int(len(data) * sample_fraction)

    # Échantillonnage aléatoire d'indices
    sampled_indices = np.random.choice(len(data), size=num_samples, replace=False)

    # Créer un nouvel ensemble de données échantillonné
    sampled_data = data[sampled_indices]

    return sampled_data

# Exemple d'utilisation
coords_file = 'E:/cours_geomatique_3eme_annee/PFE/pratique/projet1/data/val_coords.npy'
labels_file = 'E:/cours_geomatique_3eme_annee/PFE/pratique/projet1/data/val_labels.npy'

# Échantillonner les coordonnées
sampled_coords = sample_data(coords_file, sample_fraction=0.1)
np.save('/projet1/data/val_coords.npy', sampled_coords)

# Échantillonner les labels
sampled_labels = sample_data(labels_file, sample_fraction=0.1)
np.save('/projet1/data/val_labels.npy', sampled_labels)

print("Échantillonnage terminé. Fichiers sauvegardés : sample_train_coords.npy et sample_train_labels.npy")