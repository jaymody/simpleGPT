import math
import os
from typing import Optional

import lightning
import numpy as np
import toml
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from pydantic import BaseModel, ConfigDict, FilePath
from pydantic.v1.utils import deep_update
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset

from .model import GPT
from .utils import set_seed


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(BaseConfig):
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int


class DataConfig(BaseConfig):
    train_fpath: FilePath
    val_fpath: FilePath
    seq_len: int
    train_batch_size: int
    val_batch_size: int
    num_workers: int


class OptimizationConfig(BaseConfig):
    betas: list[float]
    weight_decay: float
    start_lr: float
    peak_lr: float
    end_lr: float
    warmup_steps: int
    decay_steps: int
    dropout_p: float


class Config(BaseConfig):
    include: Optional[str] = None
    seed: int
    output_dir: str
    torch_compile: bool
    tokenizer_name: str
    wandb: Optional[dict] = None
    checkpointing: dict
    model: ModelConfig
    data: DataConfig
    optimization: OptimizationConfig
    trainer: dict


def load_toml_with_include(fpath):
    with open(fpath) as f:
        data = toml.load(f)

    fpath = os.path.abspath(fpath)
    dirpath = os.path.dirname(fpath)

    relative_path = data.get("include")
    if relative_path is not None:
        assert isinstance(relative_path, str)
        included_toml_path = os.path.join(dirpath, relative_path)
        included_toml_data = load_toml_with_include(included_toml_path)
        data = deep_update(included_toml_data, data)

    return data


def cosine_with_warmup(start_lr, peak_lr, end_lr, warmup_steps, decay_steps):
    """Learning rate schedule with warmup and cosine decay.

    The learning rate starts at start_lr and linearly increases to peak_lr over
    warmup_steps steps.

    Then we decay down to end_lr for decay_steps steps using a cosine function.

    After, we just keep a constant learning rate of end_lr.

    The returned function is designed to be used with torch.optim.lr_scheduler.LambdaLR,
    that is the returned function returns the factor the lr needs to be multiplied with,
    not the lr itself at the given step.
    """

    def get_lr(step):
        if step < warmup_steps:
            ratio = start_lr / peak_lr
            t = step / warmup_steps
            return ratio + (1 - ratio) * t
        elif step < warmup_steps + decay_steps:
            ratio = end_lr / peak_lr
            t = math.cos(((step - warmup_steps) / decay_steps) * math.pi) * 0.5 + 0.5
            return ratio + (1 - ratio) * t
        else:
            return end_lr / peak_lr

    return get_lr


class LightningGPT(lightning.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump())
        self.gpt = GPT(
            **config.model.model_dump(), dropout_p=config.optimization.dropout_p
        )

    def configure_optimizers(self):
        param_dict = dict(self.named_parameters())
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if n.endswith(".weight")]
        nodecay_params = [p for n, p in param_dict.items() if not n.endswith(".weight")]
        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.optimization.weight_decay,
            },
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.optimization.peak_lr,
            betas=self.config.optimization.betas,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            cosine_with_warmup(
                start_lr=self.config.optimization.start_lr,
                peak_lr=self.config.optimization.peak_lr,
                end_lr=self.config.optimization.end_lr,
                warmup_steps=self.config.optimization.warmup_steps,
                decay_steps=self.config.optimization.decay_steps,
            ),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def training_step(self, ids, _):
        loss = self.gpt.lm_loss(ids)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, ids, _):
        loss = self.gpt.lm_loss(ids)
        self.log("val_loss", loss, prog_bar=True)


class GPTDataset(IterableDataset):
    def __init__(self, fpath, seq_len):
        super().__init__()
        self.data = np.memmap(fpath, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __iter__(self):
        return self

    def __next__(self):
        idx = np.random.randint(len(self.data) - self.seq_len)
        return self.data[idx : idx + self.seq_len].astype(np.int64)


def main(config_path: str):
    config = Config.model_validate(load_toml_with_include(config_path))

    os.makedirs(config.output_dir)

    set_seed(config.seed)

    torch.set_float32_matmul_precision("high")

    train_dataset = GPTDataset(config.data.train_fpath, config.data.seq_len)
    val_dataset = GPTDataset(config.data.val_fpath, config.data.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.val_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    model = LightningGPT(config)
    if config.torch_compile:
        model = torch.compile(model)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=os.path.join(config.output_dir, "ckpts"),
            **config.checkpointing,
        ),
    ]

    if config.wandb is not None:
        logger = WandbLogger(save_dir=config.output_dir, **config.wandb)
        logger.name
    else:
        logger = TensorBoardLogger(save_dir=config.output_dir)

    trainer = lightning.Trainer(
        **config.trainer,
        default_root_dir=config.output_dir,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
