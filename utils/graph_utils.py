import torch
from sklearn.neighbors import NearestNeighbors

def create_graph(coords, k=20):
    """
    Crée un graphe à partir des coordonnées en utilisant k voisins.

    Args:
        coords (torch.Tensor): Les coordonnées des points.
        k (int): Le nombre de voisins à considérer.

    Returns:
        edge_index (torch.Tensor): La matrice d'adjacence du graphe.
    """

    def create_graph(coords, k=20):

        nbrs = NearestNeighbors(n_neighbors=k).fit(coords.numpy())
        distances, indices = nbrs.kneighbors(coords.numpy())

        edge_index = []
        for i in range(len(coords)):
            for j in indices[i]:
                edge_index.append([i, j])

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        print("Edge index shape:", edge_index_tensor.shape)  # Ajoutez cette ligne pour vérifier
        return edge_index_tensor