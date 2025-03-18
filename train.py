# train.py

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models.dgcnn import DGCNN
from utils.data_loader import load_data
from utils.graph_utils import create_graph
from utils.metrics import print_metrics

# Hyperparamètres
NUM_CLASSES = 7  # Remplacez par le nombre de classes dans vos données
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.01

# Charger les données
train_data = load_data('data/train_coords.npy', 'data/train_labels.npy')
val_data = load_data('data/val_coords.npy', 'data/val_labels.npy')

# Créer les graphes
train_data.edge_index = create_graph(train_data.x)  # Assurez-vous que cela est fait
val_data.edge_index = create_graph(val_data.x)      # Assurez-vous que cela est fait

# Créer les DataLoader
train_loader = DataLoader([train_data], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader([val_data], batch_size=BATCH_SIZE)

# Initialiser le modèle, l'optimiseur et la fonction de perte
model = DGCNN(num_classes=NUM_CLASSES).cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Boucle d'entraînement
for epoch in range(NUM_EPOCHS):
    model.train()
    for data in train_loader:
        data = data.cuda()
        print("Data x shape:", data.x.shape)  # Vérifiez la forme de x

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            data = data.cuda()
            output = model(data)
            print_metrics(output, data.y)

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')