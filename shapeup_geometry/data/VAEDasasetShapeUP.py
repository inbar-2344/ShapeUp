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
    batch_size: int = 1
    num_workers: int = 1
    # explicit list of uids to use; if None, uids are read from {root_dir}/{split}.json
    ids_list: Optional[List[str]] = None

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


class BaseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: BaseDataModuleConfig = cfg
        start_idx = cfg.start_idx
        end_idx = cfg.end_idx
        self.split = split
        print(f"split {split}")
        if getattr(cfg, "ids_list", None) is not None:
            self.uids = list(cfg.ids_list)
        else:
            self.uids = json.load(open(f"{cfg.root_dir}/{split}.json"))
        self.uids = sorted(self.uids)[start_idx: end_idx]
        if self.cfg.const_image is not None and self.cfg.load_image and self.cfg.const_image.endswith('.json'):
            self.cfg.const_image = json.load(open(self.cfg.const_image))
            self.uids = [uid for uid in self.uids for im in self.cfg.const_image]
        # random.seed(42)        # Set seed for reproducibility
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

    # def _load_shape_from_occupancy_or_sdf(self, index: int) -> Dict[str, Any]:
    #     if self.cfg.geo_data_type == "sdf":
    #         obj_dir = pjoin(self.cfg.root_dir, "surfaces", self.uids[index])
    #         sub_dirs = os.listdir(obj_dir)
    #         if "source" in sub_dirs:
    #             is_motion = False
    #             source_surface_dir = "source"
    #         else:
    #             is_motion = True
    #             source_surface_dir = sub_dirs[0]
        
    #         source_data = np.load(pjoin(obj_dir, source_surface_dir, "pc.npz"))
    #         source_surface = source_data["fps_coarse_surface"][:, 0]
    #         source_surface[:, :3] = source_surface[:, :3] * self.cfg.scale 
    #         source_sharp_surface = source_data["fps_sharp_surface"][:, 0]
    #         source_sharp_surface[:, :3] = source_sharp_surface[:, :3] * self.cfg.scale 
    #         sub_dirs.remove(source_surface_dir)

    #         variations_surface = list()
    #         variations_sharp_surface = list()
    #         for sub_dir in sub_dirs:
    #             data = np.load(pjoin(obj_dir, sub_dir, "pc.npz"))
    #             surface = data["fps_coarse_surface"][:, 0]
    #             sharp_surface = data["fps_sharp_surface"][:, 0]
    #             surface[:, :3] = surface[:, :3] * self.cfg.scale  # target scale
    #             sharp_surface[:, :3] = sharp_surface[:, :3] * self.cfg.scale  # target scale
    #             variations_surface.append(surface)
    #             variations_sharp_surface.append(sharp_surface)
            
    #         ret = {
    #             "uid": self.uids[index].split("/")[-1],
    #             "surface": source_surface.astype(np.float32),
    #             "sharp_surface": source_sharp_surface.astype(np.float32),
    #             "objs_surface": np.stack(variations_surface, axis=0).astype(np.float32),
    #             "objs_sharp_surface": np.stack(variations_sharp_surface, axis=0).astype(np.float32),
    #             "npz_name": pjoin(source_surface_dir, "pc.npz"),
    #             "parts_indices": sub_dirs,
    #             "is_motion": is_motion
    #             }
    #         return ret

    #     else:
    #         raise NotImplementedError(
    #             f"Data type {self.cfg.geo_data_type} not implemented"
    #         )
        
    def _load_shape_from_occupancy_or_sdf(self, index: int) -> Dict[str, Any]:
        if self.cfg.geo_data_type == "sdf":
            obj_dir = pjoin(self.cfg.root_dir, "surfaces", self.uids[index])
            sub_dirs = [d for d in os.listdir(obj_dir) if os.path.isdir(pjoin(obj_dir, d))]
            motion_and_parts = False
            is_parts = False
            is_motion = False
            if "source" in sub_dirs:
                is_parts = True
            for sub_dir in sub_dirs:
                if sub_dir.startswith("frame_"):
                    is_motion = True
                    break
            motion_and_parts = is_motion and is_parts

            variations_surface = list()
            variations_sharp_surface = list()
            motion_sub_dirs = sorted([sd for sd in sub_dirs if sd.startswith("frame_")])
            parts_sub_dirs = sorted([sd for sd in sub_dirs if sd not in motion_sub_dirs])

            sub_dirs = motion_sub_dirs + parts_sub_dirs # make sure parts subdirs are first
            
            for sub_dir in sub_dirs:
                data = np.load(pjoin(obj_dir, sub_dir, "pc.npz"))
                surface = data["fps_coarse_surface"][:, 0]
                sharp_surface = data["fps_sharp_surface"][:, 0]
                surface[:, :3] = surface[:, :3] * self.cfg.scale  # target scale
                sharp_surface[:, :3] = sharp_surface[:, :3] * self.cfg.scale  # target scale
                variations_surface.append(surface)
                variations_sharp_surface.append(sharp_surface)
            
            ret = {
                "uid": self.uids[index].split("/")[-1],
                "objs_surface": np.stack(variations_surface, axis=0).astype(np.float32),
                "objs_sharp_surface": np.stack(variations_sharp_surface, axis=0).astype(np.float32),
                "parts_indices": sub_dirs,
                "motion_and_parts": motion_and_parts
                }
            return ret

        else:
            raise NotImplementedError(
                f"Data type {self.cfg.geo_data_type} not implemented"
            )
        
    
    def _load_shape_supervision_occupancy_or_sdf(self, index: int) -> Dict[str, Any]:
        # for supervision
        ret = {}
        obj_dir = pjoin(self.cfg.root_dir, "surfaces", self.uids[index])
        sub_dirs = [d for d in os.listdir(obj_dir) if os.path.isdir(pjoin(obj_dir, d))]
        if "source" in sub_dirs:
            source_surface_dir = "source"
        else:
            source_surface_dir = sub_dirs[0]
        npz_name = "pc.npz"
        source_npz_name = pjoin(obj_dir, source_surface_dir, npz_name)
        data = np.load(source_npz_name)
        ret["sharp_near_surface"] = data["sharp_near_surface"]
        ret["rand_points"] = data["rand_points"]
        ret["npz_name"] = source_npz_name
        
        return ret


    def _load_image(self, index: int) -> Dict[str, Any]:
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

        ret = {}
        if self.cfg.const_image:
            print(f"const_image: {self.cfg.const_image}")
            img_path = self.cfg.const_image[index % len(self.cfg.const_image)]
            print(img_path)
            ret["img_id"] = img_path.split("/")[-1][:-4]
            
        else:
            obj_dir = pjoin(self.cfg.root_dir, "surfaces", self.uids[index])
            base_img_name = [im for im in os.listdir(obj_dir) if im.endswith(".png")][0]
            img_path = pjoin(obj_dir, base_img_name)
        
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
                # load supervision for shape
                if self.cfg.load_geometry_supervision:
                    ret.update(self._load_shape_supervision_occupancy_or_sdf(index))
            else:
                raise NotImplementedError(
                    f"Geo data type {self.cfg.geo_data_type} not implemented"
                )

            if flip:  # random flip the input point cloud and the supervision
                for key in ret.keys():
                    if key in ["surface", "sharp_surface", "part_surface", "part_sharp_surface", 'obj_coarse_surface', 'obj_sharp_surface']:  # N x (xyz + normal)
                        ret[key][:, 0] = -ret[key][:, 0]
                        ret[key][:, 3] = -ret[key][:, 3]
                    elif key in ["rand_points"]:
                        ret[key][:, 0] = -ret[key][:, 0]


        # load image
        if self.cfg.load_image:
            ret.update(self._load_image(index))

            if flip:  # random flip the input image
                for key in ret.keys():
                    if key in ["input_image"]:  # random flip the input image
                        ret[key] = torch.flip(ret[key], [2])
                    if key in ["input_mask"]:  # random flip the input image
                        ret[key] = torch.flip(ret[key], [2])

        meta = None
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
