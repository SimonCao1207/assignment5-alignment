from collections.abc import Callable

import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class SFTDataset(Dataset):
    def __init__(
        self,
        config,
        tokenizer: PreTrainedTokenizerBase,
        data_path: str = "data/gsm8k/train.jsonl",
    ):
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        assert data_path.endswith(".jsonl")
        self.df = pd.read_json(data_path, lines=True)
        self.prompts = self.df["question"].tolist()
        self.responses = self.df["answer"].tolist()
        self.batch_size = config.batch_size

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.prompts[idx], self.responses[idx]


def make_collate_fn(tokenizer: PreTrainedTokenizerBase) -> Callable[[list[tuple[str, str]]], dict[str, Tensor]]:
    def collate_fn(batch: list[tuple[str, str]]) -> dict[str, Tensor]:
        prompts, responses = zip(*batch)
        prompts, responses = list(prompts), list(responses)
        return tokenize_prompt_and_output(prompts, responses, tokenizer)

    return collate_fn


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, Tensor]:
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
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    collate_fn = make_collate_fn(tokenizer)
    train_dataset = SFTDataset(tokenizer, "data/gsm8k/train.jsonl")
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    batch = next(iter(train_loader))
    for k, v in batch.items():
        print(k, v.shape, v.dtype)
        break
