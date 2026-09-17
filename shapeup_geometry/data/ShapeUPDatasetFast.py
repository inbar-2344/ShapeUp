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

def sample_k_points(points: np.ndarray, k: int) -> np.ndarray:
    n_points = points.shape[0]
    if k <= n_points:
        indices = np.random.choice(n_points, k, replace=False)
    else:
        indices = np.random.choice(n_points, k, replace=True)

    return points[indices].copy()


@dataclass
class BaseDataModuleConfig:
    root_dir: str = None
    validation_root_dir: str = None
    batch_size: int = 4
    num_workers: int = 8

    ################################# General argumentation #################################
    random_flip: bool = (
        False  # whether to randomly flip the input point cloud and the input images
    )

    ################################# Geometry part #################################
    load_geometry: bool = True  # whether to load geometry data
    with_sharp_data: bool = True
    geo_data_type: str = "sdf"  # occupancy, sdf
    # for occupancy or sdf supervision
    n_samples: int = 4096  # number of points in input point cloud
    noise_sigma: float = 0.0  # noise level of the input point cloud
    rotate_parts: bool = ( 
        False  # whether to rotate the parts point cloud and the supervision, for VAE aug.
    )
    scale: float = 1.0  # scale of the input point cloud and target supervision
    scale_parts: bool = ( # indices 
        False  # whether to scale the parts point cloud and the supervision, for VAE aug.
    )
    n_parts: int = 8
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
    
    const_image: Optional[Any] = None
    start_idx: int = 0 #not used
    end_idx: int = 10000 #not used 
    images_list: Optional[List[str]] = None
    motion_json_path: Optional[str] = None  # Path to train_motion.json
    motion_weight: float = 1.0  # Weight multiplier for motion samples


class BaseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: BaseDataModuleConfig = cfg
        if split in ["test", "val"]:
            self.cfg.load_geometry_supervision = True
        self.split = split
        print(f"split {split}")
        self.uids = json.load(open(f"{cfg.root_dir}/{split}.json"))
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

    def _load_shape(self, index: int) -> Dict[str, Any]:
        obj_dir = pjoin(self.cfg.root_dir, "surfaces", self.uids[index])
        npz_name = [f for f in os.listdir(obj_dir) if f.endswith('.npz')][0]
        data = np.load(pjoin(obj_dir, npz_name))
        posterior_mean = data["objs_posterior_mean"]
        posterior_std = data["objs_posterior_std"]
        is_motion_and_parts = data["motion_and_parts"]
        if is_motion_and_parts:
            value = random.randint(0, 1) # 0: motion, 1: parts
            if value == 0:
                samples = random.sample(range(3), k=2)
            else:
                samples = random.sample(range(3, len(posterior_mean)), k=2)
        else:
            samples = random.sample(range(len(posterior_mean)), k=2)
        #samples = random.choices(range(len(posterior_mean)), k=2)
        shape_posterior_mean = posterior_mean[samples[0]]
        shape_posterior_std = posterior_std[samples[0]]
        obj_posterior_mean = posterior_mean[samples[1]]
        obj_posterior_std = posterior_std[samples[1]]

        # rescale data
        ret = {
            "uid": self.uids[index].split("/")[-1],
            "shape_posterior_mean": shape_posterior_mean.astype(np.float32),
            "shape_posterior_std": shape_posterior_std.astype(np.float32),
            "obj_posterior_mean": obj_posterior_mean.astype(np.float32),
            "obj_posterior_std": obj_posterior_std.astype(np.float32),
            "gt_shape_index": samples[0],
            "input_shape_index": samples[1],
            "shapes_dirs": list(data["parts_indices"])
        }
        return ret

    def _load_shape_supervision(self, index: int, gt_shape_index: int, input_shape_index: int) -> Dict[str, Any]:
        # for supervision
        ret = {}
        obj_dir = pjoin(self.cfg.root_dir, "surfaces", self.uids[index])
        data = np.load(pjoin(obj_dir, "pc.npz"))
        gt_shape_surface = data["objs_surface"][:, gt_shape_index, :].astype(np.float32)
        input_shape_surface = data["objs_surface"][:, input_shape_index, :].astype(np.float32)
        ret["gt_shape_surface"] = gt_shape_surface 
        ret["input_shape_surface"] = input_shape_surface 
        return ret

    def _load_image_path(self, img_path: str) -> str:
        def _process_img(image, background_color=(255, 255, 255), foreground_ratio=0.9):
            alpha = image.getchannel("A")
            background = Image.new("RGBA", image.size, (*background_color, 255))
            image = Image.alpha_composite(background, image)
            image = image.crop(alpha.getbbox())

            new_size = tuple(int(dim * foreground_ratio) for dim in image.size)
            resized_image = image.resize(new_size)
            padded_image = Image.new("RGBA", image.size, (*background_color, 255))
            paste_position = (
                (image.width - resized_image.width) // 2,
                (image.height - resized_image.height) // 2,
            )
            padded_image.paste(resized_image, paste_position)

            # Expand image to 1:1
            max_dim = max(padded_image.size)
            image = Image.new("RGBA", (max_dim, max_dim), (*background_color, 255))
            paste_position = (
                (max_dim - padded_image.width) // 2,
                (max_dim - padded_image.height) // 2,
            )
            image.paste(padded_image, paste_position)
            image = image.resize((512, 512))
            return image.convert("RGB"), alpha

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
        
        return torch.from_numpy(np.array(image) / 255.0), torch.from_numpy(np.array(alpha) / 255.0).unsqueeze(0)

    def _load_image(self, index: int, gt_shape_dir: str, input_shape_dir: str) -> Dict[str, Any]:
        ret = {}
        input_shape_dir = pjoin(self.cfg.root_dir, "surfaces", self.uids[index], input_shape_dir, "mv")
        gt_shape_dir = pjoin(self.cfg.root_dir, "surfaces", self.uids[index], gt_shape_dir, "mv")
        if len(os.listdir(input_shape_dir)) == 6:
            image_ind = 0# random view index
        else:
            image_ind = random.randint(3 , len(os.listdir(input_shape_dir)) - 1) # random view index
        if image_ind <= 5:
            image_ind = 0
        input_image_name = [f for f in os.listdir(gt_shape_dir) if f.endswith(".webp")][image_ind] # input image should be takes from gt shape dir 
        gt_image_name = [f for f in os.listdir(input_shape_dir) if f.endswith(".webp")][0] # always use front view as gt, its for vis only anyway

        input_img_path = pjoin(gt_shape_dir, input_image_name)
        gt_img_path = pjoin(input_shape_dir, gt_image_name)

        if self.cfg.image_type == "rgb" or self.cfg.image_type == "normal":
            # gt_shape -> input image! 
            input_img, input_mask = self._load_image_path(input_img_path)
            gt_img, gt_mask = self._load_image_path(gt_img_path)
            ret["input_image"] = input_img
            ret["input_mask"] = input_mask
            ret["original_shape_image"] = gt_img
            ret["original_shape_mask"] = gt_mask
        else:
            raise NotImplementedError(
                f"Image type {self.cfg.image_type} not implemented"
            )

        return ret

    def _get_data(self, index):
        ret = {"uid": self.uids[index]}

        # random flip
        # flip = np.random.rand() < 0.5 if self.cfg.random_flip else False

        # load geometry
        if self.cfg.load_geometry:
            if self.cfg.geo_data_type == "occupancy" or self.cfg.geo_data_type == "sdf":
                # load shape
                ret = self._load_shape(index)
                # load supervision for shape
                if self.cfg.load_geometry_supervision:
                    ret.update(self._load_shape_supervision(index, ret["gt_shape_index"], ret["input_shape_index"]))
            else:
                raise NotImplementedError(
                    f"Geo data type {self.cfg.geo_data_type} not implemented"
                )
        # load image
        if self.cfg.load_image:
            input_shape_dir = ret["shapes_dirs"][ret["input_shape_index"]].item()
            gt_shape_dir = ret["shapes_dirs"][ret["gt_shape_index"]].item()
            ret.pop("shapes_dirs", None)
            ret.update(self._load_image(index, gt_shape_dir=gt_shape_dir, input_shape_dir=input_shape_dir))
            # if flip:  # random flip the input image
            #     for key in ret.keys():
            #         if key in ["image"]:  # random flip the input image
            #             ret[key] = torch.flip(ret[key], [2])
            #         if key in ["mask"]:  # random flip the input image
            #             ret[key] = torch.flip(ret[key], [2])

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
