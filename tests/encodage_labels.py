import numpy as np

# Charger les labels
new_labels = np.load('data_npy/labels_val.npy')  # Remplacez par le chemin de votre fichier

# Vérifier les labels avant le mapping
print("Labels avant le mapping:", np.unique(new_labels))

# Créer un dictionnaire de correspondance
label_mapping = {
    'unclassified': 0,
    'building': 1,
    'ground': 2,
    'high vegetation': 3,
    'medium vegetation': 4,
    'low vegetation': 5,
    'water': 6
}

# Mapper les labels en utilisant le dictionnaire
mapped_labels = np.vectorize(label_mapping.get)(new_labels)

# Vérifier les labels après le mapping
print("Labels après le mapping:", np.unique(mapped_labels))

# Sauvegarder les labels modifiés
np.save('data_npy/mapped_labels_val.npy', mapped_labels)

print("Les labels ont été mappés et sauvegardés avec succès.")