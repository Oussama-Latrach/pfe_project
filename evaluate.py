# evaluate.py

import torch
from torch_geometric.loader import DataLoader
from models.dgcnn import DGCNN
from utils.data_loader import load_data
from utils.graph_utils import create_graph
from utils.metrics import print_metrics

# Hyperparamètres
NUM_CLASSES = 7  # Remplacez par le nombre de classes dans vos données
BATCH_SIZE = 32
#######

# Charger les données de test
test_data = load_data('data/test_coords.npy', 'data/test_labels.npy')

# Créer le graphe
test_data.edge_index = create_graph(test_data.x)

# Créer le DataLoader
test_loader = DataLoader([test_data], batch_size=BATCH_SIZE)

# Initialiser le modèle
model = DGCNN(num_classes=NUM_CLASSES).cuda()
model.load_state_dict(torch.load('dgcnn_model.pth'))  # Charger le modèle pré-entraîné
model.eval()

# Évaluation
with torch.no_grad():
    for data in test_loader:
        data = data.cuda()
        output = model(data)
        print_metrics(output, data.y)