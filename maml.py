import torch
from collections import OrderedDict
from torch.func import functional_call

def _named_params_dict(model: torch.nn.Module) -> OrderedDict:
    return OrderedDict((name, p) for name, p in model.named_parameters())

def _clone_detached_requires_grad(params: OrderedDict) -> OrderedDict:
    return OrderedDict((n, t.detach().clone().requires_grad_()) for n, t in params.items())

def fomaml_meta_grads(
    model: torch.nn.Module,
    loss_fn,
    xs: torch.Tensor,
    ys: torch.Tensor,
    xq: torch.Tensor,
    yq: torch.Tensor,
    lr_inner: float,
    inner_steps: int = 1,
) -> tuple[OrderedDict, torch.Tensor]:
    """
    Compute First-Order MAML meta-gradients for a single client.

    Returns (grads_by_name, query_loss).
    """
    # Initialize params from the model as independent leaves
    params = _clone_detached_requires_grad(_named_params_dict(model))

    # Inner-loop adaptation on support set (first-order: no higher-order graphs)
    for _ in range(max(1, int(inner_steps))):
        logits_s = functional_call(model, params, (xs,))
        loss_s = loss_fn(logits_s, ys)
        grads_s = torch.autograd.grad(loss_s, params.values(), create_graph=False)
        params = OrderedDict(
            (name, (param - lr_inner * g).detach().clone().requires_grad_())
            for (name, param), g in zip(params.items(), grads_s)
        )

    # Query forward with adapted params
    logits_q = functional_call(model, params, (xq,))
    loss_q = loss_fn(logits_q, yq)

    # First-order meta-gradients: d L_q / d theta' (treat theta' as independent of theta)
    grads_q = torch.autograd.grad(loss_q, params.values(), create_graph=False)
    grads_by_name = OrderedDict((name, g) for (name, _), g in zip(params.items(), grads_q))
    return grads_by_name, loss_q.detach()
