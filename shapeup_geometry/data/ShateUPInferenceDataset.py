import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import random
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from os.path import join as pjoin
from shapeup_geometry.utils.typing import *


@dataclass
class BaseDataModuleConfig:
    root_dir: str = None
    validation_root_dir: str = None
    batch_size: int = 1
    num_workers: int = 1

    ################################# General argumentation #################################
    random_flip: bool = (
        False  # whether to randomly flip the input point cloud and the input images
    )

    ################################# Geometry part #################################
    load_geometry: bool = True  # whether to load geometry data
    with_sharp_data: bool = False
    geo_data_type: str = "sdf"  # occupancy, sdf
    # for occupancy or sdf supervision
    n_samples: int = 4096  # number of points in input point cloud
    upsample_ratio: int = 1  # upsample ratio for input point cloud
    sampling_strategy: Optional[str] = (
        "random"  # sampling strategy for input point cloud
    )
    scale: float = 1.0  # scale of the input point cloud and target supervision
    noise_sigma: float = 0.0  # noise level of the input point cloud
    rotate_points: bool = (
        False  # whether to rotate the input point cloud and the supervision, for VAE aug.
    )
    load_geometry_supervision: bool = False  # whether to load supervision
    supervision_type: str = "sdf"  # occupancy, sdf, tsdf, tsdf_w_surface
    n_supervision: int = 10000  # number of points in supervision
    tsdf_threshold: float = (
        0.01  # threshold for truncating sdf values, used when input is sdf
    )

    ################################# Image part #################################
    load_image: bool = False  # whether to load images
    image_type: str = "rgb"  # rgb, normal, rgb_or_normal
    image_file_type: str = "png"  # png, jpeg
    image_type_ratio: float = (
        1.0  # ratio of rgb for each dataset when image_type is "rgb_or_normal"
    )
    crop_image: bool = True  # whether to crop the input image
    random_color_jitter: bool = (
        False  # whether to randomly color jitter the input images
    )
    random_rotate: bool = (
        False  # whether to randomly rotate the input images, default [-10 deg, 10 deg]
    )
    random_mask: bool = False  # whether to add random mask to the input image
    background_color: Tuple[int, int, int] = field(
        default_factory=lambda: (255, 255, 255)
    )
    idx: Optional[List[int]] = None  # index of the image to load
    n_views: int = 1  # number of views
    foreground_ratio: Optional[float] = 0.90

    ################################# Caption part #################################
    start_idx: int = 0
    end_idx: int = 10000
    const_image: Optional[Any] = None
    images_list: Optional[List[str]] = None
    motion_json_path: Optional[str] = None  # Path to train_motion.json
    motion_weight: float = 1.0  # Weight multiplier for motion samples


class BaseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: BaseDataModuleConfig = cfg
        start_idx = cfg.start_idx
        end_idx = cfg.end_idx
        self.split = split
        print(f"split {split}")
        self.uids = json.load(open(f"{cfg.root_dir}/{split}.json"))
        self.uids = sorted(self.uids)[start_idx: end_idx]
        if self.cfg.validation_root_dir is None:
            self.cfg.validation_root_dir = "data/editing_examples"
        
        self.images_list = []
        full_uids = []
        for uid in self.uids:
            img_dir = pjoin(self.cfg.validation_root_dir, uid, "edited_mv")
            images_list = [im for im in os.listdir(img_dir)]
            self.images_list.extend(images_list)
            full_uids.extend([uid] * len(images_list))
        self.uids = full_uids



        random.seed(10)        # Set seed for reproducibility
        # random.shuffle(self.uids)
        print(f"Loaded {len(self.uids)} {split} uids")

        # add ColorJitter transforms for input images
        if self.cfg.random_color_jitter:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            )

        # add RandomRotation transforms for input images
        if self.cfg.random_rotate:
            self.rotate = transforms.RandomRotation(
                degrees=10, fill=(*self.cfg.background_color, 0.0)
            )  # by default 10 deg

    def __len__(self):
        return len(self.uids)

        
    def _load_shape_from_occupancy_or_sdf(self, index: int) -> Dict[str, Any]:
        if self.cfg.geo_data_type == "sdf":
            obj_dir = pjoin(self.cfg.validation_root_dir, self.uids[index])
            sub_dir = "source"
            data = np.load(pjoin(obj_dir, sub_dir, "pc.npz"))
            surface = data["fps_coarse_surface"][:, 0]
            sharp_surface = data["fps_sharp_surface"][:, 0]
            surface[:, :3] = surface[:, :3] * self.cfg.scale  # target scale
            sharp_surface[:, :3] = sharp_surface[:, :3] * self.cfg.scale  # target scale
            
            ret = {
                "uid": self.uids[index].split("/")[-1],
                "objs_surface": surface[None].astype(np.float32),
                "objs_sharp_surface": sharp_surface[None].astype(np.float32),
                }
            return ret

        else:
            raise NotImplementedError(
                f"Data type {self.cfg.geo_data_type} not implemented"
            )
        


    def _load_image(self, index: int) -> Dict[str, Any]:
        def _process_img(image, background_color=(255, 255, 255), foreground_ratio=0.9):
            # The alpha channel goes through the same crop/resize/pad as the image.
            # Returning the *original* alpha leaves input_mask at the source
            # resolution, so a batch of differently sized edit images cannot be
            # collated (torch.stack on mismatched masks).
            alpha = image.getchannel("A")
            bbox = alpha.getbbox()
            background = Image.new("RGBA", image.size, (*background_color, 255))
            image = Image.alpha_composite(background, image)
            image = image.crop(bbox)
            alpha = alpha.crop(bbox)

            new_size = tuple(int(dim * foreground_ratio) for dim in image.size)
            resized_image = image.resize(new_size)
            resized_alpha = alpha.resize(new_size)
            padded_image = Image.new("RGBA", image.size, (*background_color, 255))
            padded_alpha = Image.new("L", image.size, 0)
            paste_position = (
                (image.width - resized_image.width) // 2,
                (image.height - resized_image.height) // 2,
            )
            padded_image.paste(resized_image, paste_position)
            padded_alpha.paste(resized_alpha, paste_position)

            # Expand image to 1:1
            max_dim = max(padded_image.size)
            image = Image.new("RGBA", (max_dim, max_dim), (*background_color, 255))
            alpha = Image.new("L", (max_dim, max_dim), 0)
            paste_position = (
                (max_dim - padded_image.width) // 2,
                (max_dim - padded_image.height) // 2,
            )
            image.paste(padded_image, paste_position)
            alpha.paste(padded_alpha, paste_position)
            image = image.resize((512, 512))
            alpha = alpha.resize((512, 512))
            return image.convert("RGB"), alpha

        ret = {}
        img_dir = pjoin(self.cfg.validation_root_dir, self.uids[index], "edited_mv")
        # img_name = [im for im in os.listdir(img_dir)][0]
        img_name = self.images_list[index]
        img_path = pjoin(img_dir, img_name)
        if self.cfg.image_type == "rgb" or self.cfg.image_type == "normal":
            assert (
                self.cfg.n_views == 1
            ), "Only single view is supported for single image"
            image = Image.open(img_path).copy()

            # add random color jitter
            if self.cfg.random_color_jitter:
                rgb = self.color_jitter(image.convert("RGB"))
                image = Image.merge("RGBA", (*rgb.split(), image.getchannel("A")))

            # add random rotation
            if self.cfg.random_rotate:
                image = self.rotate(image)

            # add crop
            if self.cfg.crop_image:
                background_color = (
                    torch.randint(0, 256, (3,))
                    if self.cfg.background_color is None
                    else torch.as_tensor(self.cfg.background_color)
                )
                image, alpha = _process_img(
                    image, background_color, self.cfg.foreground_ratio
                )
            else:
                alpha = image.getchannel("A")
                background = Image.new("RGBA", image.size, background_color)
                image = Image.alpha_composite(background, image).convert("RGB")

            ret["input_image"] = torch.from_numpy(np.array(image) / 255.0)
            ret["input_mask"] = torch.from_numpy(np.array(alpha) / 255.0).unsqueeze(0)
            ret["input_image_name"] = img_name
        else:
            raise NotImplementedError(
                f"Image type {self.cfg.image_type} not implemented"
            )

        return ret
    def _get_data(self, index):
        ret = {"uid": self.uids[index]}

        # random flip
        flip = np.random.rand() < 0.5 if self.cfg.random_flip else False

        # load geometry
        if self.cfg.load_geometry:
            if self.cfg.geo_data_type == "occupancy" or self.cfg.geo_data_type == "sdf":
                # load shape
                ret = self._load_shape_from_occupancy_or_sdf(index)

            else:
                raise NotImplementedError(
                    f"Geo data type {self.cfg.geo_data_type} not implemented"
                )

        # load image
        if self.cfg.load_image:
            ret.update(self._load_image(index))

        return ret

    def __getitem__(self, index):
        try:
            return self._get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map

        return torch.utils.data.default_collate(batch)
