# utils/data_loader.py

import numpy as np
import torch
from torch_geometric.data import Data

def load_data(coords_file, labels_file):
    """
    Charge les coordonnées et les labels à partir des fichiers .npy.

    Args:
        coords_file (str): Chemin vers le fichier des coordonnées.
        labels_file (str): Chemin vers le fichier des labels.

    Returns:
        Data: Un objet Data contenant les coordonnées et les labels.
    """
    coords = np.load(coords_file)
    labels = np.load(labels_file)

    coords_tensor = torch.tensor(coords, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return Data(x=coords_tensor, y=labels_tensor)

