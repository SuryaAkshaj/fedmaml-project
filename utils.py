import os
import math
import copy
import torch
import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def get_device(use_gpu=True):
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

def clone_model_state(model):
    return copy.deepcopy(model.state_dict())

def apply_state(model, state_dict):
    model.load_state_dict(state_dict)

def state_dict_to_vector(state_dict):
    return torch.cat([p.flatten() for p in state_dict.values()])

def vector_to_state_dict(vector, reference_state):
    new_state = OrderedDict()
    idx = 0
    for k, v in reference_state.items():
        numel = v.numel()
        new_state[k] = vector[idx:idx+numel].view_as(v)
        idx += numel
    return new_state

def add_dp_noise(update_state, std=0.0, device="cpu"):
    if std <= 0:
        return update_state
    noisy = OrderedDict()
    for k, v in update_state.items():
        noise = torch.randn_like(v, device=device) * std
        noisy[k] = v + noise
    return noisy

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses, preds, trues = [], [], []
    crit = torch.nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        losses.append(loss.item())
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = accuracy_score(y_true, y_pred)
    return float(np.mean(losses)), float(acc)

def average_states(states, weights=None):
    """
    states: list of state_dicts
    weights: list of floats (sum to 1). If None, uniform.
    """
    if weights is None:
        weights = [1.0/len(states)] * len(states)
    avg = OrderedDict()
    for k in states[0]:
        avg[k] = sum(w * s[k] for w, s in zip(weights, states))
    return avg

def plot_curves(curves, save_path, title, xlabel="Round", ylabel="Value"):
    plt.figure(figsize=(6,4))
    for label, values in curves.items():
        plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def split_train_val_from_loader(loader, val_fraction=0.2):
    # Build a single dataset from loader.dataset indices
    if hasattr(loader.dataset, 'indices'):
        indices = loader.dataset.indices
        n = len(indices)
        n_val = int(n * val_fraction)
        return indices[n_val:], indices[:n_val]
    else:
        n = len(loader.dataset)
        n_val = int(n * val_fraction)
        return list(range(n_val, n)), list(range(0, n_val))
