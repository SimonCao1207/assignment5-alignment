import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_pretrained(model, tokenizer, output_dir):
    """Save the model and tokenizer to the specified directory."""
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


model_name = "Qwen/Qwen2.5-Math-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
