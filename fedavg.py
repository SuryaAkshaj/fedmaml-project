import torch
from torch.optim import SGD, Adam
from collections import OrderedDict
from utils import clone_model_state, apply_state, average_states, add_dp_noise

def local_train_one_client(model, loader, lr=0.01, local_epochs=1, optimizer_name="sgd", device="cpu"):
    model.train()
    crit = torch.nn.CrossEntropyLoss()
    if optimizer_name.lower() == "adam":
        opt = Adam(model.parameters(), lr=lr)
    else:
        opt = SGD(model.parameters(), lr=lr, momentum=0.9)

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

def fedavg_round(global_model, client_loaders, selected_idx, lr=0.01, local_epochs=1, optimizer="sgd",
                 dp_noise_std=0.0, device="cpu"):
    global_state = clone_model_state(global_model)
    client_states = []
    for i in selected_idx:
        # init with global
        apply_state(global_model, global_state)
        local_train_one_client(global_model, client_loaders[i], lr=lr, local_epochs=local_epochs,
                               optimizer_name=optimizer, device=device)
        local_state = clone_model_state(global_model)
        # delta = local - global
        delta = OrderedDict({k: local_state[k] - global_state[k] for k in global_state})
        # optional DP noise
        delta = add_dp_noise(delta, std=dp_noise_std, device=device)
        # reconstruct local state = global + noisy delta
        noisy_local_state = OrderedDict({k: global_state[k] + delta[k] for k in global_state})
        client_states.append(noisy_local_state)

    new_global = average_states(client_states)
    apply_state(global_model, new_global)
