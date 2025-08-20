# Personalized Federated Learning with Meta-Learning (FedMAML vs FedAvg)

This repo implements a complete federated learning system where:
- Data are partitioned non-IID across clients via a Dirichlet prior.
- Baseline: **Federated Averaging (FedAvg)**.
- Proposed: **Federated MAML (First-Order MAML)** integrated inside the FL rounds.
- Optional **Differential Privacy**: add Gaussian noise to client updates (FedAvg) or to meta-gradients (FedMAML).
- Evaluations: global accuracy + **personalized accuracy** after quick adaptation on each client.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
