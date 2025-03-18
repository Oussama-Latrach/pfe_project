# utils/metrics.py

import torch

def accuracy(preds, labels):
    """
    Calcule la précision du modèle.

    Args:
        preds (torch.Tensor): Les prédictions du modèle.
        labels (torch.Tensor): Les labels réels.

    Returns:
        float: La précision du modèle.
    """
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def f1_score(preds, labels, average='macro'):
    """
    Calcule le score F1 du modèle.

    Args:
        preds (torch.Tensor): Les prédictions du modèle.
        labels (torch.Tensor): Les labels réels.
        average (str): La méthode de calcul du score F1 ('macro', 'micro', 'weighted').

    Returns:
        float: Le score F1 du modèle.
    """
    from sklearn.metrics import f1_score as sklearn_f1

    _, predicted = torch.max(preds, 1)
    return sklearn_f1(labels.cpu().numpy(), predicted.cpu().numpy(), average=average)

def print_metrics(preds, labels):
    """
    Affiche les métriques de performance.

    Args:
        preds (torch.Tensor): Les prédictions du modèle.
        labels (torch.Tensor): Les labels réels.
    """
    acc = accuracy(preds, labels)
    f1 = f1_score(preds, labels)
    print(f'Accuracy: {acc:.4f}, F1 Score: {f1:.4f}')