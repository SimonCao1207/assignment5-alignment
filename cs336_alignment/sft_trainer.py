from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.sft_dataset import SFTDataset, make_collate_fn
from cs336_alignment.utils import load_pretrained, masked_normalize


@dataclass
class TrainConfig:
    # Model
    learning_rate: float = 6e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-1

    # Data
    batch_size: int = 8


class SFTTrainer:
    def __init__(
        self,
        config: TrainConfig,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self._build_dataloader(train_dataset, val_dataset)

        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay
        )

    def _build_dataloader(self, train_ds, val_ds):
        self.train_dataset, self.val_dataset = train_ds, val_ds
        collate_fn = make_collate_fn(self.tokenizer)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def train(self):
        pass


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
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    model, tokenizer = load_pretrained(model_name)
    config = TrainConfig()
    train_ds = SFTDataset(tokenizer, "data/gsm8k/train.jsonl", config.batch_size)
    val_ds = SFTDataset(tokenizer, "data/gsm8k/test.jsonl", config.batch_size)
    trainer = SFTTrainer(config, tokenizer, model, train_ds, val_ds)
    trainer.train()
