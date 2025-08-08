import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM


def compute_entropy(logits: Float[Tensor, "B S V"]) -> Float[Tensor, "B S"]:
    probs = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log(probs + 1e-10)  # Adding a small value to avoid log(0)
    entropy = -torch.sum(probs * log_probabilities, dim=-1)
    return entropy


def get_reponse_log_probs(
    model: torch.nn.Module,
    input_ids: Float[Tensor, "B S"],
    labels: Float[Tensor, "B S"],
    return_token_entropy: bool = False,
) -> dict[str, Float[Tensor, "B S"]]:
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    softmax_logits: Float[Tensor, "B S V"] = torch.log_softmax(logits, dim=-1)
    log_probs: Float[Tensor, "B S"] = softmax_logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": entropy}

    return {"log_probs": log_probs}


if __name__ == "__main__":
    # Example usage
    logits = torch.randn(2, 5, 10)  # Batch size 2, sequence length 5, vocabulary size 10
    entropy = compute_entropy(logits)
    print("Entropy:", entropy)

    model_name = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_ids = torch.randint(0, 10, (2, 5))  # Random input ids
    labels = torch.randint(0, 10, (2, 5))  # Random labels
    response_log_probs = get_reponse_log_probs(model, input_ids, labels)
    print("Response Log Probabilities:", response_log_probs["log_probs"])
