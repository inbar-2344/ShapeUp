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
from torch.utils.data import DataLoader, WeightedRandomSampler
from shapeup_geometry import register
from shapeup_geometry.utils.typing import *
from shapeup_geometry.utils.config import parse_structured
import numpy as np
from streaming import StreamingDataLoader
from .ShapeUPDatasetFast import BaseDataModuleConfig, BaseDataset
from .ShateUPInferenceDataset import BaseDataset as TestBaseDataset 
import random
from os.path import join as pjoin
import traceback

@dataclass
class ShapeupDataModuleConfig(BaseDataModuleConfig):
    pass

class ShaepNetDataset(BaseDataset):
    pass

class ShapeupInferenceDataset(TestBaseDataset):
    pass


@register("ShapeUP-Fast-datamodule")
class ShaepNetDataModule(pl.LightningDataModule):
    cfg: ShapeupDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ShapeupDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ShaepNetDataset(self.cfg, "train")
            self.train_sampler = None
            if self.cfg.motion_json_path:
                motion_uids = set(json.load(open(self.cfg.motion_json_path)))

                weights = []
                for uid in self.train_dataset.uids:
                    if uid in motion_uids:
                        weights.append(self.cfg.motion_weight)
                    else:
                        weights.append(1.0)

                self.train_sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True
                )
                n_motion = sum(1 for uid in self.train_dataset.uids if uid in motion_uids)
                print(f"Weighted sampling enabled: {n_motion} motion samples ({100*n_motion/len(weights):.1f}%) with weight {self.cfg.motion_weight}")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ShapeupInferenceDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ShapeupInferenceDataset(self.cfg, "test")

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
        if self.trainer and self.trainer.state.fn != 'fit':
            print("train_dataloader() called outside of fit stage!")
            return torch.utils.data.DataLoader([])
        sampler = getattr(self, 'train_sampler', None)
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        print("in val dataloader")
        return self.general_loader(
            self.val_dataset,
            batch_size=self.cfg.batch_size//2,
            collate_fn=self.val_dataset.collate,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        print("in test dataloader")
        return self.general_loader(
            self.test_dataset,
            batch_size=self.cfg.batch_size//2,
            collate_fn=self.test_dataset.collate,
            num_workers=self.cfg.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        print("in predict dataloader")
        return self.general_loader(self.test_dataset, batch_size=self.cfg.batch_size)
    