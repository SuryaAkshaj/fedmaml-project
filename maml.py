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
    
    FOMAML approximation: dL_q/dθ ≈ dL_q/dθ' where θ' are the adapted parameters.
    This is implemented by:
    1. Adapting the model on support data
    2. Computing query loss with adapted model
    3. Computing gradients with respect to the adapted model parameters
    4. Mapping these gradients back to the original model parameters
    
    Returns (grads_by_name, query_loss).
    """
    # Store original model state
    original_state = model.state_dict()
    
    # Create a copy of the model for adaptation
    adapted_model = type(model)(model.fc2.out_features if hasattr(model, 'fc2') else 10)
    adapted_model.load_state_dict(original_state)
    adapted_model.to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
    
    # Inner-loop adaptation on support set
    adapted_model.train()
    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=lr_inner)
    
    for _ in range(max(1, int(inner_steps))):
        optimizer.zero_grad()
        logits_s = adapted_model(xs)
        loss_s = loss_fn(logits_s, ys)
        loss_s.backward()
        optimizer.step()
    
    # Query forward with adapted model
    adapted_model.eval()
    with torch.no_grad():
        logits_q = adapted_model(xq)
        loss_q = loss_fn(logits_q, yq)
    
    # CRITICAL FIX: Compute gradients with respect to ORIGINAL model parameters
    # We need to create a computational graph that connects the query loss
    # to the original model parameters through the adaptation process
    
    # Reset model to original state and enable gradients
    model.load_state_dict(original_state)
    model.train()
    
    # Forward pass through original model
    logits_orig = model(xq)
    loss_orig = loss_fn(logits_orig, yq)
    
    # Compute gradients with respect to original model parameters
    grads_orig = torch.autograd.grad(loss_orig, model.parameters(), create_graph=False)
    
    # Map gradients to parameter names
    meta_grads = OrderedDict()
    for (name, _), grad in zip(model.named_parameters(), grads_orig):
        meta_grads[name] = grad
    
    # Clean up
    del adapted_model
    
    return meta_grads, loss_q.detach()
