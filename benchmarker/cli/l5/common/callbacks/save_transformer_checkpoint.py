from pathlib import Path
from typing import Union
import pytorch_lightning as pl

from benchmarker.cli.l5.common.pl_modules.base_lightning_module import BaseLightningModule


class SaveTransformerCheckpoint(pl.Callback):
    def __init__(self, save_path: Union[str, Path]):
        self.save_path = save_path

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: BaseLightningModule):
        pl_module.model.config.save_step = pl_module.step_count
        pl_module.model.save_pretrained(self.save_path)
        pl_module.tokenizer.save_pretrained(self.save_path)
