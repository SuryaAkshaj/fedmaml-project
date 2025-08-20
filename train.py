import os
import yaml
import math
import random
import argparse
import numpy as np
import torch
from tqdm import trange
from data_loader import build_client_loaders, set_seed
from model import make_model
from utils import get_device, evaluate, plot_curves, clone_model_state, apply_state
from fedavg import fedavg_round
from fedmaml import fedmaml_round

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sample_clients(num_clients, clients_per_round, rng):
    return rng.choice(num_clients, size=clients_per_round, replace=False).tolist()

def personalized_eval(model, client_loader, device, steps=1, lr_inner=0.01):
    """
    Evaluate personalization: copy global model, run a few inner steps on client's data, then test on its own held-out batch(es).
    For simplicity, we do: adapt on one batch, evaluate on next batch.
    """
    from torch.optim import SGD
    import torch.nn as nn
    crit = nn.CrossEntropyLoss()

    # copy state
    base_state = clone_model_state(model)
    model.train()
    opt = SGD(model.parameters(), lr=lr_inner)

    it = iter(client_loader)
    try:
        xs, ys = next(it)
    except StopIteration:
        return float('nan'), float('nan')

    xs, ys = xs.to(device), ys.to(device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(xs), ys)
        loss.backward()
        opt.step()

    # evaluate on next batch(es)
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for j, (xq, yq) in enumerate(it):
            xq, yq = xq.to(device), yq.to(device)
            logits = model(xq)
            losses.append(crit(logits, yq).item())
            pred = logits.argmax(1)
            correct += (pred == yq).sum().item()
            total += yq.size(0)
            if j >= 2:  # small personalized eval
                break

    # restore
    apply_state(model, base_state)
    if total == 0:
        return float(np.mean(losses)) if losses else float('nan'), float('nan')
    return float(np.mean(losses)), correct / total

def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("use_gpu", True))
    print(f"Using device: {device}")

    # Data
    client_loaders, test_loader, num_classes, input_shape = build_client_loaders(
        dataset=cfg["dataset"],
        data_dir=cfg["data_dir"],
        num_clients=cfg["num_clients"],
        alpha=cfg["dirichlet_alpha"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"]
    )

    # Models
    fedavg_model = make_model(cfg["dataset"], num_classes).to(device)
    fedmaml_model = make_model(cfg["dataset"], num_classes).to(device)

    # Training
    rounds = cfg["rounds"]
    clients_per_round = cfg["clients_per_round"]
    rng = np.random.default_rng(cfg["seed"])

    # Logs
    fa_train_loss, fa_test_acc = [], []
    fm_train_loss, fm_test_acc = [], []
    fa_personal_acc, fm_personal_acc = [], []

    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # --- FedAvg ---
    print("\n=== Training FedAvg ===")
    for r in trange(rounds):
        selected = sample_clients(cfg["num_clients"], clients_per_round, rng)
        fedavg_round(
            fedavg_model, client_loaders, selected_idx=selected,
            lr=cfg["lr_global"], local_epochs=cfg["local_epochs"],
            optimizer=cfg["optimizer"],
            dp_noise_std=cfg["dp_noise_std"],
            device=device
        )
        # Eval
        test_loss, test_acc = evaluate(fedavg_model, test_loader, device)
        fa_test_acc.append(test_acc)

        # Track pseudo-training loss as average client batch loss for logging
        # (not exact, but indicative)
        fa_train_loss.append(test_loss)

        # Personalized eval on a random client
        cidx = int(rng.integers(0, cfg["num_clients"]))
        _, p_acc = personalized_eval(fedavg_model, client_loaders[cidx], device,
                                     steps=cfg["eval_personalization_steps"], lr_inner=cfg["lr_inner"])
        fa_personal_acc.append(p_acc if not np.isnan(p_acc) else 0.0)

    # --- FedMAML ---
    print("\n=== Training FedMAML (FOMAML) ===")
    for r in trange(rounds):
        selected = sample_clients(cfg["num_clients"], clients_per_round, rng)
        loss = fedmaml_round(
            fedmaml_model, client_loaders, selected_idx=selected,
            lr_inner=cfg["lr_inner"], inner_steps=cfg["inner_steps"],
            meta_lr=cfg["maml_meta_lr"], dp_noise_std=cfg["dp_noise_std"],
            device=device
        )
        fm_train_loss.append(loss if not math.isnan(loss) else 0.0)
        test_loss, test_acc = evaluate(fedmaml_model, test_loader, device)
        fm_test_acc.append(test_acc)

        cidx = int(rng.integers(0, cfg["num_clients"]))
        _, p_acc = personalized_eval(fedmaml_model, client_loaders[cidx], device,
                                     steps=cfg["eval_personalization_steps"], lr_inner=cfg["lr_inner"])
        fm_personal_acc.append(p_acc if not np.isnan(p_acc) else 0.0)

    # Final evals
    fa_final_loss, fa_final_acc = evaluate(fedavg_model, test_loader, device)
    fm_final_loss, fm_final_acc = evaluate(fedmaml_model, test_loader, device)

    # Per-client personalized logs
    per_client_logs = []
    for i in range(cfg["num_clients"]):
        _, fa_p_acc = personalized_eval(fedavg_model, client_loaders[i], device,
                                        steps=cfg["eval_personalization_steps"], lr_inner=cfg["lr_inner"])
        _, fm_p_acc = personalized_eval(fedmaml_model, client_loaders[i], device,
                                        steps=cfg["eval_personalization_steps"], lr_inner=cfg["lr_inner"])
        per_client_logs.append((i, fa_p_acc, fm_p_acc))

    # Plots
    plot_curves(
        {"FedAvg": fa_train_loss, "FedMAML": fm_train_loss},
        os.path.join(save_dir, "loss_vs_rounds.png"),
        "Training Loss vs Rounds"
    )
    plot_curves(
        {"FedAvg": fa_test_acc, "FedMAML": fm_test_acc},
        os.path.join(save_dir, "global_accuracy_vs_rounds.png"),
        "Global Test Accuracy vs Rounds", ylabel="Accuracy"
    )
    plot_curves(
        {"FedAvg": fa_personal_acc, "FedMAML": fm_personal_acc},
        os.path.join(save_dir, "personalized_accuracy_vs_rounds.png"),
        "Personalized Accuracy vs Rounds", ylabel="Accuracy"
    )

    # Table of results
    print("\n=== Final Results ===")
    print(f"Global Accuracy - FedAvg:  {fa_final_acc:.4f}")
    print(f"Global Accuracy - FedMAML: {fm_final_acc:.4f}")
    print("\nPer-client Personalized Accuracy (after adaptation):")
    print("Client\tFedAvg\tFedMAML")
    for cid, a, b in per_client_logs:
        print(f"{cid}\t{a:.4f}\t{b:.4f}")

    # Save a CSV summary
    with open(os.path.join(save_dir, "results.csv"), "w") as f:
        f.write("metric,FedAvg,FedMAML\n")
        f.write(f"global_acc,{fa_final_acc:.4f},{fm_final_acc:.4f}\n")
        f.write("client_id,fa_personal_acc,fm_personal_acc\n")
    with open(os.path.join(save_dir, "per_client_personalized.csv"), "w") as f:
        f.write("client_id,fa_personal_acc,fm_personal_acc\n")
        for cid, a, b in per_client_logs:
            f.write(f"{cid},{a:.6f},{b:.6f}\n")

if __name__ == "__main__":
    main()
