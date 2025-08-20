import torch
from torch.optim import Adam
from collections import OrderedDict
from utils import clone_model_state, apply_state, add_dp_noise
from maml import fomaml_meta_grads

def fedmaml_round(global_model, client_loaders, selected_idx, lr_inner=0.01, inner_steps=1,
                  meta_lr=0.001, dp_noise_std=0.0, device="cpu"):
    """
    One FL round with FOMAML:
      For each selected client:
        - Start from global.
        - Sample support and query minibatches from client's data.
        - Do K inner updates on support to get adapted params (implicit).
        - Compute query loss; backprop to global params (first-order).
      Server aggregates gradients and applies meta update.
    """
    crit = torch.nn.CrossEntropyLoss()
    global_model.train()
    global_state = clone_model_state(global_model)

    total_clients = len(selected_idx)
    loss_sum = 0.0

    # Accumulate per-parameter grads in a dict
    accumulated_grads: OrderedDict[str, torch.Tensor] = OrderedDict(
        (name, torch.zeros_like(param)) for name, param in global_model.named_parameters()
    )

    for i in selected_idx:
        # reset to global
        apply_state(global_model, global_state)

        # build two mini-batches (support, query)
        loader = client_loaders[i]
        it = iter(loader)
        try:
            xs, ys = next(it)
        except StopIteration:
            continue
        try:
            xq, yq = next(it)
        except StopIteration:
            xq, yq = xs.clone(), ys.clone()

        xs, ys = xs.to(device), ys.to(device)
        xq, yq = xq.to(device), yq.to(device)

        client_grads, q_loss = fomaml_meta_grads(
            global_model, crit, xs, ys, xq, yq, lr_inner=lr_inner, inner_steps=inner_steps
        )
        loss_sum += float(q_loss.item())

        # Accumulate
        for name, p in global_model.named_parameters():
            g = client_grads.get(name)
            if g is not None:
                accumulated_grads[name].add_(g)

    # Average and optionally add DP noise
    for name, p in global_model.named_parameters():
        g = accumulated_grads[name]
        g.div_(max(1, total_clients))
        if dp_noise_std > 0:
            g.add_(torch.randn_like(g) * dp_noise_std)

    # Apply meta update: theta <- theta - meta_lr * g
    with torch.no_grad():
        for name, p in global_model.named_parameters():
            p.add_( -meta_lr * accumulated_grads[name] )

    return loss_sum / max(1, total_clients)
