import os
import yaml
import math
import random
import argparse
import numpy as np
import torch
import logging
from tqdm import trange
from data_loader import build_client_loaders, set_seed
from model import make_model
from utils import get_device, evaluate, plot_curves, clone_model_state, apply_state
from fedavg import fedavg_round
from fedmaml import fedmaml_round

def setup_logging(save_dir):
    """Setup logging configuration"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate configuration parameters
    required_keys = [
        "dataset", "data_dir", "num_clients", "dirichlet_alpha", 
        "clients_per_round", "rounds", "local_epochs", "batch_size",
        "test_batch_size", "optimizer", "lr_global", "lr_inner", 
        "inner_steps", "maml_meta_lr", "eval_personalization_steps",
        "dp_noise_std", "seed", "use_gpu", "save_dir"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate specific parameter ranges
    if config["num_clients"] <= 0:
        raise ValueError("num_clients must be positive")
    if config["clients_per_round"] > config["num_clients"]:
        raise ValueError("clients_per_round cannot exceed num_clients")
    if config["dirichlet_alpha"] <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    if config["rounds"] <= 0:
        raise ValueError("rounds must be positive")
    if config["local_epochs"] <= 0:
        raise ValueError("local_epochs must be positive")
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if config["lr_global"] <= 0:
        raise ValueError("lr_global must be positive")
    if config["lr_inner"] <= 0:
        raise ValueError("lr_inner must be positive")
    if config["maml_meta_lr"] <= 0:
        raise ValueError("maml_meta_lr must be positive")
    if config["inner_steps"] <= 0:
        raise ValueError("inner_steps must be positive")
    if config["dp_noise_std"] < 0:
        raise ValueError("dp_noise_std must be non-negative")
    
    # Validate dataset
    if config["dataset"].upper() not in ["MNIST", "CIFAR10"]:
        raise ValueError("dataset must be either 'MNIST' or 'CIFAR10'")
    
    # Validate optimizer
    if config["optimizer"].lower() not in ["sgd", "adam"]:
        raise ValueError("optimizer must be either 'sgd' or 'adam'")
    
    logging.info("Configuration validation passed!")
    return config

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

    # Convert loader to list to avoid iterator issues
    data_batches = list(client_loader)
    if len(data_batches) < 2:
        return float('nan'), float('nan')

    # Use first batch for adaptation
    xs, ys = data_batches[0]
    xs, ys = xs.to(device), ys.to(device)
    
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(xs), ys)
        loss.backward()
        opt.step()

    # evaluate on remaining batches for better assessment
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for j, (xq, yq) in enumerate(data_batches[1:], 1):
            xq, yq = xq.to(device), yq.to(device)
            logits = model(xq)
            losses.append(crit(logits, yq).item())
            pred = logits.argmax(1)
            correct += (pred == yq).sum().item()
            total += yq.size(0)
            # Use more batches for evaluation (up to 5)
            if j >= 5:
                break

    # restore
    apply_state(model, base_state)
    if total == 0:
        return float(np.mean(losses)) if losses else float('nan'), float('nan')
    return float(np.mean(losses)), correct / total

def check_convergence(accuracies, window_size=10, tolerance=0.001):
    """
    Check if training has converged based on accuracy improvement.
    
    Args:
        accuracies: List of accuracy values
        window_size: Number of recent rounds to consider
        tolerance: Minimum improvement threshold
    
    Returns:
        bool: True if converged, False otherwise
    """
    if len(accuracies) < window_size:
        return False
    
    recent_accs = accuracies[-window_size:]
    improvement = max(recent_accs) - min(recent_accs)
    
    return improvement < tolerance

def main():
    try:
        args = parse_args()
        cfg = load_config(args.config)

        set_seed(cfg.get("seed", 42))
        device = get_device(cfg.get("use_gpu", True))
        
        # Setup logging
        logger = setup_logging(cfg["save_dir"])
        logger.info(f"Using device: {device}")
        logger.info(f"Configuration: {cfg}")

        # Data
        logger.info("Building client data loaders...")
        try:
            client_loaders, test_loader, num_classes, input_shape = build_client_loaders(
                dataset=cfg["dataset"],
                data_dir=cfg["data_dir"],
                num_clients=cfg["num_clients"],
                alpha=cfg["dirichlet_alpha"],
                batch_size=cfg["batch_size"],
                seed=cfg["seed"]
            )
            logger.info(f"Data loaded: {len(client_loaders)} clients, {num_classes} classes")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

        # Models
        try:
            fedavg_model = make_model(cfg["dataset"], num_classes).to(device)
            fedmaml_model = make_model(cfg["dataset"], num_classes).to(device)
            logger.info(f"Models initialized: {cfg['dataset']} with {num_classes} classes")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

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
        logger.info("=== Starting FedAvg Training ===")
        fa_converged = False
        for r in trange(rounds):
            try:
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
                
                if r % 10 == 0:
                    logger.info(f"FedAvg Round {r}: Test Acc={test_acc:.4f}, Personal Acc={p_acc:.4f}")
                
                # Check convergence
                if r >= 20 and check_convergence(fa_test_acc, window_size=10, tolerance=0.001):
                    logger.info(f"FedAvg converged at round {r}")
                    fa_converged = True
                    break
                    
            except Exception as e:
                logger.error(f"Error in FedAvg round {r}: {e}")
                # Continue training but log the error
                fa_test_acc.append(0.0)
                fa_train_loss.append(0.0)
                fa_personal_acc.append(0.0)

        # --- FedMAML ---
        logger.info("=== Starting FedMAML Training ===")
        fm_converged = False
        for r in trange(rounds):
            try:
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
                
                if r % 10 == 0:
                    logger.info(f"FedMAML Round {r}: Loss={loss:.4f}, Test Acc={test_acc:.4f}, Personal Acc={p_acc:.4f}")
                
                # Check convergence
                if r >= 20 and check_convergence(fm_test_acc, window_size=10, tolerance=0.001):
                    logger.info(f"FedMAML converged at round {r}")
                    fm_converged = True
                    break
                    
            except Exception as e:
                logger.error(f"Error in FedMAML round {r}: {e}")
                # Continue training but log the error
                fm_train_loss.append(0.0)
                fm_test_acc.append(0.0)
                fm_personal_acc.append(0.0)

        # Log convergence status
        logger.info(f"FedAvg convergence: {'Yes' if fa_converged else 'No'}")
        logger.info(f"FedMAML convergence: {'Yes' if fm_converged else 'No'}")

        # Final evals
        try:
            fa_final_loss, fa_final_acc = evaluate(fedavg_model, test_loader, device)
            fm_final_loss, fm_final_acc = evaluate(fedmaml_model, test_loader, device)
        except Exception as e:
            logger.error(f"Failed to perform final evaluation: {e}")
            fa_final_loss, fa_final_acc = 0.0, 0.0
            fm_final_loss, fm_final_acc = 0.0, 0.0

        # Per-client personalized logs
        logger.info("Computing per-client personalized performance...")
        per_client_logs = []
        for i in range(cfg["num_clients"]):
            try:
                _, fa_p_acc = personalized_eval(fedavg_model, client_loaders[i], device,
                                                steps=cfg["eval_personalization_steps"], lr_inner=cfg["lr_inner"])
                _, fm_p_acc = personalized_eval(fedmaml_model, client_loaders[i], device,
                                                steps=cfg["eval_personalization_steps"], lr_inner=cfg["lr_inner"])
                per_client_logs.append((i, fa_p_acc, fm_p_acc))
            except Exception as e:
                logger.error(f"Failed to evaluate client {i}: {e}")
                per_client_logs.append((i, 0.0, 0.0))

        # Plots
        logger.info("Generating visualization plots...")
        try:
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
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

        # Table of results
        logger.info("=== Final Results ===")
        logger.info(f"Global Accuracy - FedAvg:  {fa_final_acc:.4f}")
        logger.info(f"Global Accuracy - FedMAML: {fm_final_acc:.4f}")
        logger.info("Per-client Personalized Accuracy (after adaptation):")
        logger.info("Client\tFedAvg\tFedMAML")
        for cid, a, b in per_client_logs:
            logger.info(f"{cid}\t{a:.4f}\t{b:.4f}")

        # Save a CSV summary
        try:
            with open(os.path.join(save_dir, "results.csv"), "w") as f:
                f.write("metric,FedAvg,FedMAML\n")
                f.write(f"global_acc,{fa_final_acc:.4f},{fm_final_acc:.4f}\n")
                f.write("client_id,fa_personal_acc,fm_personal_acc\n")
            with open(os.path.join(save_dir, "per_client_personalized.csv"), "w") as f:
                f.write("client_id,fa_personal_acc,fm_personal_acc\n")
                for cid, a, b in per_client_logs:
                    f.write(f"{cid},{a:.6f},{b:.6f}\n")
        except Exception as e:
            logger.error(f"Failed to save CSV files: {e}")
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {save_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
