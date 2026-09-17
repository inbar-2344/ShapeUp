import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from shapeup_geometry import register
from shapeup_geometry.utils.typing import *
from shapeup_geometry.utils.config import parse_structured
import numpy as np
from streaming import StreamingDataLoader
from .VAEDasasetShapeUP import BaseDataModuleConfig, BaseDataset
import random
from os.path import join as pjoin
import traceback

@dataclass
class ShaepNetDataModuleConfig(BaseDataModuleConfig):
    pass


class ShaepNetDataset(BaseDataset):
    pass


@register("VAE-datamodule")
class ShaepNetDataModule(pl.LightningDataModule):
    cfg: ShaepNetDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ShaepNetDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ShaepNetDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ShaepNetDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ShaepNetDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(
        self, dataset, batch_size, collate_fn=None, num_workers=0
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        print("in test dataloader")
        if self.trainer and self.trainer.state.fn != 'fit':
            print("train_dataloader() called outside of fit stage!")
            return torch.utils.data.DataLoader([])
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        print("in val dataloader")
        return self.general_loader(self.val_dataset, batch_size=self.cfg.batch_size)

    def test_dataloader(self) -> DataLoader:
        print("in test dataloader")
        return self.general_loader(self.test_dataset, batch_size=self.cfg.batch_size)

    def predict_dataloader(self) -> DataLoader:
        print("in predict dataloader")
        return self.general_loader(self.test_dataset, batch_size=self.cfg.batch_size)
    