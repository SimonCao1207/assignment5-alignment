import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_file: str, tokenizer, config):
        self.parquet_file = parquet_file
        self.tokenizer: PreTrainedTokenizerBase = tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    """
    Let prompt_and_output_lens be a list containing the lengths of
    the tokenized prompt and output strings. Then the returned dictionary should have the
    following keys:

    input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the tokenized prompt and output strings, with the final token sliced off.
    labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): shifted input ids, i.e., the input ids without the first token.
    response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): a mask on the response tokens in the labels.
    """

    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    assert isinstance(prompt_ids, list) and isinstance(response_ids, list)
    input_ids_list = []
    response_mask_list = []

    for p_ids, r_ids in zip(prompt_ids, response_ids):
        full_ids = p_ids + r_ids  # concat
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        mask = torch.zeros(len(full_ids) - 1, dtype=torch.long)
        mask[len(p_ids) - 1 :] = 1
        input_ids_list.append(input_ids)
        response_mask_list.append(mask)
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)  # type: ignore
    response_mask = pad_sequence(response_mask_list, batch_first=True, padding_value=0)
    labels = input_ids.clone()

    input_ids = input_ids[:, :-1]  # remove the last token
    labels = labels[:, 1:]  # remove the first token

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


if __name__ == "__main__":
    prompt_strs = [
        "Hello, world!",
        "This is a test.",
        "This is another test.",
    ]
    output_strs = [
        "Hello, world!",
        "This is a test.",
        "This is another test.",
    ]
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    result = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
