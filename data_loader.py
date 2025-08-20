import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_transforms(dataset):
    if dataset.upper() == "MNIST":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset.upper() == "CIFAR10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    else:
        raise ValueError("Unsupported dataset")
    return tfm

def load_dataset(dataset="MNIST", data_dir="./data"):
    tfm = _get_transforms(dataset)
    if dataset.upper() == "MNIST":
        train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
        num_classes = 10
        input_shape = (1, 28, 28)
    else:
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
        num_classes = 10
        input_shape = (3, 32, 32)
    return train, test, num_classes, input_shape

def dirichlet_non_iid_splits(labels, num_clients, alpha=0.3, min_size=10, seed=42):
    """
    Split indices into non-IID partitions via Dirichlet distribution over labels.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    n_classes = len(np.unique(labels))
    idx_by_class = [np.where(labels == y)[0] for y in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]
    # sample class proportions for each client
    class_client_props = rng.dirichlet([alpha] * num_clients, n_classes)
    # assign indices per class
    for c in range(n_classes):
        idx_c = idx_by_class[c]
        splits = (np.cumsum((class_client_props[c] / class_client_props[c].sum()) * len(idx_c))).astype(int)
        splits = np.clip(splits, 0, len(idx_c))
        prev = 0
        chunks = []
        for s in splits:
            chunks.append(idx_c[prev:s])
            prev = s
        # chunks length may be != num_clients due to rounding; pad/merge
        while len(chunks) < num_clients:
            chunks.append(np.array([], dtype=int))
        chunks = chunks[:num_clients]
        for i in range(num_clients):
            client_indices[i].extend(chunks[i].tolist())

    # ensure min size
    for i in range(num_clients):
        if len(client_indices[i]) < min_size:
            # steal from the largest
            sizes = [len(ci) for ci in client_indices]
            j = int(np.argmax(sizes))
            need = min_size - len(client_indices[i])
            take = client_indices[j][:need]
            client_indices[i].extend(take)
            client_indices[j] = client_indices[j][need:]

    # shuffle
    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices

def build_client_loaders(dataset="MNIST", data_dir="./data", num_clients=10,
                         alpha=0.3, batch_size=64, seed=42):
    set_seed(seed)
    train, test, num_classes, input_shape = load_dataset(dataset, data_dir)
    labels = [y for _, y in train]
    splits = dirichlet_non_iid_splits(labels, num_clients=num_clients, alpha=alpha, seed=seed)
    client_loaders = []
    for idxs in splits:
        subset = Subset(train, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)
        client_loaders.append(loader)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader, num_classes, input_shape
