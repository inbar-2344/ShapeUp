import json
import os
import time
import random
from dataclasses import dataclass, field
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
print(f"OPENCV_IO_ENABLE_OPENEXR: {os.environ['OPENCV_IO_ENABLE_OPENEXR']}")
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..utils.config import parse_structured
from ..utils.geometry import (
    get_plucker_embeds_from_cameras,
    get_plucker_embeds_from_cameras_ortho,
    get_position_map_from_depth,
    get_position_map_from_depth_ortho,
)
from ..utils.typing import *
from os.path import join as pjoin


def worker_init_fn(worker_id):
    """Initialize each DataLoader worker with required environment variables"""
    import os
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


@dataclass
class MultiviewDataModuleConfig:
    root_dir: Any = ""
    scene_list: Any = ""
    image_suffix: str = "webp"
    background_color: Union[str, float] = "gray"
    image_modality: str = "render"
    num_views: int = 1

    prompt_db_path: Optional[str] = None
    return_prompt: bool = False
    use_empty_prompt: bool = False
    prompt_prefix: Optional[Any] = None
    return_one_prompt: bool = True

    projection_type: str = "ORTHO"

    # source conditions
    source_image_modality: Any = "position"
    use_camera_space_normal: bool = False
    position_offset: float = 0.5
    position_scale: float = 1.0
    plucker_offset: float = 1.0
    plucker_scale: float = 2.0

    # reference image
    reference_image_modality: str = "render"
    reference_augment_resolutions: Optional[List[int]] = None
    reference_mask_aug: bool = False

    repeat: int = 1  # for debugging purpose

    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None

    height: int = 768
    width: int = 768

    batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 16

    motion_samples_path: Optional[str] = None
    motion_oversample_factor: float = 1.0  # 1.0 = no oversampling, 5.0 = 5x more likely


class MultiviewDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: MultiviewDataModuleConfig = cfg
        self.split = split
        self.all_scenes = json.load(open(pjoin(self.cfg.root_dir, self.split + ".json")))

        if self.cfg.prompt_db_path is not None:
            self.prompt_db = json.load(open(self.cfg.prompt_db_path))
        else:
            self.prompt_db = None

    def __len__(self):
        return len(self.all_scenes)

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif bg_color == "random_gray":
            bg_color = random.uniform(0.3, 0.7)
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, list) or isinstance(bg_color, tuple):
            bg_color = np.array(bg_color, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    

    def load_image(
        self,
        image: Union[str, Image.Image],
        height: int,
        width: int,
        background_color: torch.Tensor,
        rescale: bool = False,
        mask_aug: bool = False,
    ):
        if isinstance(image, str):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    image = Image.open(image)
                    break
                except OSError as e:
                    if attempt < max_retries - 1:
                        time.sleep(min(2.0, 0.2 * (2 ** attempt)) + random.random() * 0.1) # backoff
                    else:
                        raise OSError(f"Failed to load {image} after {max_retries} attempts: {e}")

            # image = Image.open(image)

        image = image.resize((width, height))
        image = torch.from_numpy(np.array(image)).float() / 255.0

        if mask_aug:
            alpha = image[:, :, 3]  # Extract alpha channel
            h, w = alpha.shape
            y_indices, x_indices = torch.where(alpha > 0.5)
            if len(y_indices) > 0 and len(x_indices) > 0:
                idx = torch.randint(len(y_indices), (1,)).item()
                y_center = y_indices[idx].item()
                x_center = x_indices[idx].item()
                mask_h = random.randint(h // 8, h // 4)
                mask_w = random.randint(w // 8, w // 4)

                y1 = max(0, y_center - mask_h // 2)
                y2 = min(h, y_center + mask_h // 2)
                x1 = max(0, x_center - mask_w // 2)
                x2 = min(w, x_center + mask_w // 2)

                alpha[y1:y2, x1:x2] = 0.0
                image[:, :, 3] = alpha

        image = image[:, :, :3] * image[:, :, 3:4] + background_color * (
            1 - image[:, :, 3:4]
        )
        if rescale:
            image = image * 2.0 - 1.0
        return image

    def load_normal_image(
        self,
        path,
        height,
        width,
        background_color,
        camera_space: bool = False,
        c2w: Optional[torch.FloatTensor] = None,
    ):
        image = Image.open(path).resize((width, height), resample=Image.NEAREST)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        if image.shape[2] == 4:
            alpha = image[:, :, 3:4]
            image = image[:, :, :3]
            if camera_space:
                w2c = torch.linalg.inv(c2w)[:3, :3]
                image = (
                    F.normalize(((image * 2 - 1)[:, :, None, :] * w2c).sum(-1), dim=-1)
                    * 0.5
                    + 0.5
                )
            image = image * alpha + background_color * (1 - alpha)
        else:
            image = image[:, :, :3]
        return image

    def load_depth(self, path, height, width):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depth = torch.from_numpy(depth[..., 0:1]).float()
        mask = torch.ones_like(depth)
        mask[depth > 1000.0] = 0.0  # depth = 65535 is the invalid value
        depth[~(mask > 0.5)] = 0.0
        return depth, mask

    def retrieve_prompt(self, scene_dir):
        assert self.prompt_db is not None
        source_id = os.path.basename(scene_dir)
        return self.prompt_db.get(source_id, "")

    def __getitem__(self, index):
        try:
            return self._getitem_impl(index)
        except Exception as e:
            print(f"Warning: Failed to load sample {index} ({self.all_scenes[index]}): {e}. Trying another sample.")
            # Return a random different sample
            new_index = random.randint(0, len(self) - 1)
            max_retries = 10
            for _ in range(max_retries):
                if new_index != index:
                    try:
                        return self._getitem_impl(new_index)
                    except Exception:
                        pass
                new_index = random.randint(0, len(self) - 1)
            raise RuntimeError(f"Failed to load any valid sample after {max_retries} retries")

    def _getitem_impl(self, index):
        background_color = torch.as_tensor(self.get_bg_color(self.cfg.background_color))

        if self.split == "train":
            scene_dir = pjoin(self.cfg.root_dir, "surfaces", self.all_scenes[index])
            # randomely choose which variation of the object is src and which is edited
            all_variations = [d for d in os.listdir(scene_dir) if os.path.isdir(pjoin(scene_dir, d))]
            src_dir, edited_dir = random.sample(all_variations, 2) # wo replacemnts, consider changing 
            src_dir = pjoin(scene_dir, src_dir)
            edited_dir = pjoin(scene_dir, edited_dir)
            
        else:
            scene_dir = pjoin("data/shape_diffusion/shapeup/validation_data_tex", self.all_scenes[index])
            src_dir = pjoin(scene_dir, "source")
            edited_dir = pjoin(scene_dir, "edited")

        num_views = self.cfg.num_views

        # target multi-view edited images
        edited_image_paths = [
            pjoin(
                edited_dir,"mv",f
            )
            for f in os.listdir(pjoin(edited_dir, "mv")) # all view besided the orthogonal mv
        ]
        # sort edited_image_paths by the index of the view
        edited_image_paths.sort(key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
        if self.split == "train":
            edited_mv_images = [
                self.load_image(
                    p,
                    height=self.cfg.height,
                    width=self.cfg.width,
                    background_color=background_color,
                )
                for p in edited_image_paths[:num_views]
            ] # take only the orthogonal mv views
            edited_mv_images = torch.stack(edited_mv_images, dim=0).permute(0, 3, 1, 2)
            views_names = [os.path.basename(p) for p in edited_image_paths[:num_views]]

        else:
            edited_mv_images = torch.zeros(num_views, 3, self.cfg.height, self.cfg.width) # just so collate wont fail
            views_names = ["render_0000.webp", "render_0001.webp", "render_0002.webp", "render_0003.webp", "render_0004.webp", "render_0005.webp"]

        with open(pjoin(edited_dir, "meta.json")) as f:
            meta = json.load(f)
        name2loc = {view_name: loc for loc, view_name in zip(meta["locations"], views_names)}

        # camera
        c2w = [
            torch.as_tensor(name2loc[name]["transform_matrix"])
            for name in views_names
        ]
        c2w = torch.stack(c2w, dim=0)

        if self.cfg.projection_type == "PERSP":
            camera_angle_x = (
                meta.get("camera_angle_x", None)
                or meta["locations"][0]["camera_angle_x"]
            )
            focal_length = 0.5 * self.cfg.width / np.tan(0.5 * camera_angle_x)
            intrinsics = (
                torch.as_tensor(
                    [
                        [focal_length, 0.0, 0.5 * self.cfg.width],
                        [0.0, focal_length, 0.5 * self.cfg.height],
                        [0.0, 0.0, 1.0],
                    ]
                )
                .unsqueeze(0)
                .float()
                .repeat(num_views, 1, 1)
            )
        elif self.cfg.projection_type == "ORTHO":
            ortho_scale = (
                meta.get("ortho_scale", None) or meta["locations"][0]["ortho_scale"]
            )

        # source conditions (of edited object)
        source_image_modality = self.cfg.source_image_modality
        if isinstance(source_image_modality, str):
            source_image_modality = [source_image_modality]
        source_images = []
        for modality in source_image_modality:
            if modality == "position":
                depth_paths = [pjoin(edited_dir, "depth", f) for f in os.listdir(pjoin(edited_dir, "depth"))]
                depth_paths.sort(key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
                # if positions are provides as .exr files, convert to depth maps
                if depth_paths[0].endswith(".exr"):
                    depth_masks = [
                        self.load_depth(
                            f,
                            self.cfg.height,
                            self.cfg.width,
                        )
                        for f in depth_paths[:num_views]
                    ]
                    depths = torch.stack([d for d, _ in depth_masks])
                    masks = torch.stack([m for _, m in depth_masks])
                    c2w_ = c2w.clone()
                    c2w_[:, :, 1:3] *= -1

                    if self.cfg.projection_type == "PERSP":
                        position_maps = get_position_map_from_depth(
                            depths,
                            masks,
                            intrinsics,
                            c2w_,
                            image_wh=(self.cfg.width, self.cfg.height),
                        )
                    elif self.cfg.projection_type == "ORTHO":
                        position_maps = get_position_map_from_depth_ortho(
                            depths,
                            masks,
                            c2w_,
                            ortho_scale,
                            image_wh=(self.cfg.width, self.cfg.height),
                        )
                    position_maps = (
                        (position_maps + self.cfg.position_offset) / self.cfg.position_scale
                    ).clamp(0.0, 1.0)

                else: # use load_noraml which can handle png depth images
                    position_maps = [
                        self.load_normal_image(
                            f,
                            height=self.cfg.height,
                            width=self.cfg.width,
                            background_color=background_color,
                            camera_space=self.cfg.use_camera_space_normal,
                            c2w=c,
                        )
                        for c, f in zip(c2w, depth_paths[:num_views])
                    ]
                    position_maps = torch.stack(position_maps, dim=0)
                    
                source_images.append(position_maps)
            elif modality == "normal":
                normal_paths = [
                    pjoin(edited_dir, "normal", f)
                    for f in os.listdir(pjoin(edited_dir, "normal"))
                ]
                normal_paths.sort(key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
                normal_maps = [
                    self.load_normal_image(
                        f,
                        height=self.cfg.height,
                        width=self.cfg.width,
                        background_color=background_color,
                        camera_space=self.cfg.use_camera_space_normal,
                        c2w=c,
                    )
                    for c, f in zip(c2w, normal_paths[:num_views])
                ]
                source_images.append(torch.stack(normal_maps, dim=0))
            elif modality == "plucker":
                if self.cfg.projection_type == "ORTHO":
                    plucker_embed = get_plucker_embeds_from_cameras_ortho(
                        c2w, [ortho_scale] * len(c2w), self.cfg.width
                    )
                elif self.cfg.projection_type == "PERSP":
                    plucker_embed = get_plucker_embeds_from_cameras(
                        c2w, [camera_angle_x] * len(c2w), self.cfg.width
                    )
                else:
                    raise NotImplementedError
                plucker_embed = plucker_embed.permute(0, 2, 3, 1)
                plucker_embed = (
                    (plucker_embed + self.cfg.plucker_offset) / self.cfg.plucker_scale
                ).clamp(0.0, 1.0)
                source_images.append(plucker_embed)
            else:
                raise NotImplementedError
        source_images = torch.cat(source_images, dim=-1).permute(0, 3, 1, 2)
        rv = {"rgb": edited_mv_images, "c2w": c2w, "source_rgb": source_images}

        num_images = num_views
        # prompt
        if self.cfg.return_prompt:
            if self.cfg.use_empty_prompt:
                prompt = ""
            else:
                prompt = self.retrieve_prompt(scene_dir)
            prompts = [prompt] * num_images

            if self.cfg.prompt_prefix is not None:
                prompt_prefix = self.cfg.prompt_prefix
                if isinstance(prompt_prefix, str):
                    prompt_prefix = [prompt_prefix] * num_images

                for i, prompt in enumerate(prompts):
                    prompts[i] = f"{prompt_prefix[i]} {prompt}"

            if self.cfg.return_one_prompt:
                rv.update({"prompts": prompts[0]})
            else:
                rv.update({"prompts": prompts})

        # reference image, also chosen from edited object dir 
        # work around in case no additional views exist (always use front view in this case)
        if len(edited_image_paths) <= num_views:
            reference_image_path = edited_image_paths[0]
        else:
            front_view_path = edited_image_paths[0]
            front_view_choice = [front_view_path] * 3
            reference_image_path = random.choice(front_view_choice + edited_image_paths[num_views:])# choose random view as reference, not from the orthogonal mv except front view

        if self.cfg.reference_augment_resolutions is None:
            reference_image = self.load_image(
                reference_image_path,
                height=self.cfg.height,
                width=self.cfg.width,
                background_color=background_color,
                mask_aug=self.cfg.reference_mask_aug,
            ).permute(2, 0, 1)
            rv.update({"reference_rgb": reference_image})
        else:
            random_resolution = random.choice(
                self.cfg.reference_augment_resolutions
            )
            reference_image_ = Image.open(reference_image_path).resize(
                (random_resolution, random_resolution)
            )
            reference_image = self.load_image(
                reference_image_,
                height=self.cfg.height,
                width=self.cfg.width,
                background_color=background_color,
                mask_aug=self.cfg.reference_mask_aug,
            ).permute(2, 0, 1)
            rv.update({"reference_rgb": reference_image})
    
        # load rgb mv of src object for conditioning
        src_image_paths = [
            pjoin(
                src_dir, "mv", f
            )
            for f in os.listdir(pjoin(src_dir, "mv")) # take only the orthogonal mv views
        ]
        # sort src_image_paths by the index of the view
        src_image_paths.sort(key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
        src_image_paths = src_image_paths[:num_views]
        src_images = [
            self.load_image(
                p,
                height=self.cfg.height,
                width=self.cfg.width,
                background_color=background_color,
            )
            for p in src_image_paths
        ]
        src_images = torch.stack(src_images, dim=0).permute(0, 3, 1, 2)
        rv.update({"src_rgb": src_images, "uid": scene_dir.split("/")[-1]})
        return rv

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        pack = lambda t: t.view(-1, *t.shape[2:])

        indices = list(range(self.cfg.num_views))
        num_views = len(indices)

        for k in batch.keys():
            if k in ["rgb", "source_rgb", "c2w", "src_rgb"]:
                batch[k] = batch[k][:, indices]
                batch[k] = pack(batch[k])
        for k in ["prompts"]:
            if not self.cfg.return_one_prompt:
                batch[k] = [item for pair in zip(*batch[k]) for item in pair]

        batch.update(
            {
                "num_views": num_views,
                # For SDXL
                "original_size": (self.cfg.height, self.cfg.width),
                "target_size": (self.cfg.height, self.cfg.width),
                "crops_coords_top_left": (0, 0),
            }
        )
        return batch


class MultiviewDataModule(pl.LightningDataModule):
    cfg: MultiviewDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiviewDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiviewDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiviewDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        if self.cfg.motion_samples_path is not None and self.cfg.motion_oversample_factor > 1.0:
            motion_ids = set(json.load(open(self.cfg.motion_samples_path)))
            weights = []
            for scene_id in self.train_dataset.all_scenes:
                if scene_id in motion_ids:
                    weights.append(self.cfg.motion_oversample_factor)
                else:
                    weights.append(1.0)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False  # sampler handles shuffling
            print(f"Using weighted sampling: {len(motion_ids)} motion samples with {self.cfg.motion_oversample_factor}x weight")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
            worker_init_fn=worker_init_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
