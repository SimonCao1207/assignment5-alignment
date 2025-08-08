import torch
from jaxtyping import Float
from torch import Tensor


def compute_entropy(logits: Float[Tensor, "B S V"]) -> Float[Tensor, "B S"]:
    """
    Computes the entropy of the logits.

    Args:
        logits (torch.Tensor): The logits tensor of shape (batch_size, seq_len, vocab_size).

    Returns:
        torch.Tensor: The entropy for each next-token prediction shape (batch_size, seq_len).
    """
    probs = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log(probs + 1e-10)  # Adding a small value to avoid log(0)
    entropy = -torch.sum(probs * log_probabilities, dim=-1)
    return entropy
