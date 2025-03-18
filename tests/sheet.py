import numpy as np

# Charger le fichier .npy
file_path = 'E:/cours_geomatique_3eme_annee/PFE/pratique/essaie1/labels_train.npy'  # Remplace par le chemin de ton fichier .npy
data = np.load(file_path)

# Afficher les 5 premières lignes
print(data[:5])  # Affiche les 5 premières lignes
