import numpy as np
from sklearn.model_selection import train_test_split

# Charger les données
normalized_points = np.load('normalized_points.npy')  # Remplacez par le chemin de votre fichier
new_labels = np.load('new_labels.npy')  # Remplacez par le chemin de votre fichier

# Vérifier les dimensions des données
print("Shape of normalized points:", normalized_points.shape)
print("Shape of new labels:", new_labels.shape)

# Échantillonnage des données pour obtenir 1 000 000 de points
total_sample_size = 1000000
if normalized_points.shape[0] > total_sample_size:
    indices = np.random.choice(normalized_points.shape[0], total_sample_size, replace=False)
    normalized_points = normalized_points[indices]
    new_labels = new_labels[indices]

# Diviser les données en train et test
points_train, points_test, labels_train, labels_test = train_test_split(normalized_points, new_labels, test_size=0.2, random_state=42)

# Diviser l'ensemble d'entraînement en train et validation
points_train, points_val, labels_train, labels_val = train_test_split(points_train, labels_train, test_size=0.25, random_state=42)  # 0.25 de 0.8 = 0.2 du total

# Afficher les tailles des ensembles
print(f'Taille de l\'ensemble d\'entraînement: {points_train.shape[0]}')
print(f'Taille de l\'ensemble de validation: {points_val.shape[0]}')
print(f'Taille de l\'ensemble de test: {points_test.shape[0]}')

# Sauvegarder les ensembles dans des fichiers .npy
np.save('points_train.npy', points_train)
np.save('labels_train.npy', labels_train)
np.save('points_val.npy', points_val)
np.save('labels_val.npy', labels_val)
np.save('points_test.npy', points_test)
np.save('labels_test.npy', labels_test)

print("Les ensembles de données ont été sauvegardés avec succès.")