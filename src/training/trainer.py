"""
Entraînement et prédiction du classificateur de marque (PyTorch).

L'architecture et le prétraitement viennent de src.inference.loader : le modèle
entraîné est, par construction, celui que l'application sait servir.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.inference.loader import (
    BRAND_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_brand_network,
    build_brand_transform,
)

logger = logging.getLogger(__name__)

Item = Tuple[str, int]   # (chemin de l'image, indice de la classe)


def _train_transform():
    from torchvision import transforms

    # Augmentations légères, puis exactement la normalisation de l'inférence
    return transforms.Compose([
        transforms.Resize((BRAND_INPUT_SIZE, BRAND_INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _loader(items: Sequence[Item], transform, batch_size: int, shuffle: bool):
    import torch
    from PIL import Image

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            path, target = items[i]
            with Image.open(path) as img:
                return transform(img.convert("RGB")), target

    return torch.utils.data.DataLoader(_Dataset(), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _network(state_dict, num_classes: int, device: str):
    network = build_brand_network(num_classes)
    network.load_state_dict(state_dict)
    return network.to(device)


def train(state_dict, classes: Sequence[str], items: Sequence[Item], *, epochs: int = 3, lr: float = 1e-4,
          batch_size: int = 16, freeze_backbone: bool = True, device: str = "cpu",
          seed: int = 42) -> Tuple[Dict, List[Dict[str, float]]]:
    """
    Affine le modèle (initialisé par `state_dict`) sur `items`.

    freeze_backbone : n'entraîne que le dernier bloc et la tête de classification.
    Adapté au CPU et aux petits jeux de données (moins de surapprentissage).
    → (nouveaux poids, historique par époque)
    """
    import torch

    torch.manual_seed(seed)
    network = _network(state_dict, len(classes), device)
    if freeze_backbone:
        for param in network.features.parameters():
            param.requires_grad = False
        for param in network.features[-1].parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW((p for p in network.parameters() if p.requires_grad), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    loader = _loader(items, _train_transform(), batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        network.train()
        total_loss, correct, seen = 0.0, 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(targets)
            correct += int((outputs.argmax(dim=1) == targets).sum())
            seen += len(targets)
        epoch_metrics = {"train_loss": total_loss / max(seen, 1), "train_accuracy": correct / max(seen, 1)}
        history.append(epoch_metrics)
        logger.info("Époque %d/%d : %s", epoch + 1, epochs, epoch_metrics)

    return {k: v.detach().cpu() for k, v in network.state_dict().items()}, history


def predict(state_dict, classes: Sequence[str], items: Sequence[Item], *, batch_size: int = 32,
            device: str = "cpu") -> np.ndarray:
    """Probabilités (n, nb_classes) sur `items`, avec le prétraitement exact de l'inférence."""
    import torch
    import torch.nn.functional as F

    if not items:
        return np.zeros((0, len(classes)))
    network = _network(state_dict, len(classes), device).eval()
    outputs = []
    with torch.no_grad():
        for inputs, _ in _loader(items, build_brand_transform(), batch_size, shuffle=False):
            outputs.append(F.softmax(network(inputs.to(device)), dim=1).cpu().numpy())
    return np.concatenate(outputs)
