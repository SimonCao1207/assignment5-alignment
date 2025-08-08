import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.utils import masked_normalize


def save_pretrained(model, tokenizer, output_dir):
    """Save the model and tokenizer to the specified directory."""
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


def load_pretrained(model_name):
    """Load the model and tokenizer from the specified model name."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def sft_microbatch_train_step(
    policy_log_probs: Float[Tensor, "B S"],
    response_mask: Float[Tensor, "B S"],
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    metadata = {}
    norm_nll: Float[Tensor, " B"] = masked_normalize(-policy_log_probs, response_mask, normalize_constant, dim=1)
    loss = norm_nll.mean() / gradient_accumulation_steps
    loss.backward()
    return (loss.detach(), metadata)


if __name__ == "__main__":
    policy_log_probs = torch.tensor(
        [
            [1.9269, 1.4873, 0.9007, -2.1055, -0.7581, 1.0783, 0.8008, 1.6806, 0.3559, -0.6866],
            [-0.4934, 0.2415, -0.2316, 0.0418, -0.2516, 0.8599, -0.3097, -0.3957, 0.8034, -0.6216],
        ],
        requires_grad=True,
    )

    response_mask = torch.tensor(
        [
            [True, True, False, True, False, True, False, True, True, False],
            [True, True, True, True, True, False, True, True, False, True],
        ]
    )

    gradient_accumulation_steps = 2
    normalize_constant = 42.0
    loss, metadata = sft_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=normalize_constant,
    )
