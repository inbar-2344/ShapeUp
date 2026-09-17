# Some parts of this file are refer to Hugging Face Diffusers library.
import os
import json
import time
import warnings
from typing import Callable, List, Optional, Union, Dict, Any
import PIL.Image
import trimesh
import rembg
from tqdm import tqdm
import yaml
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.loaders import (
    FluxIPAdapterMixin,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin
)
from .pipeline_utils import (
    TransformerDiffusionMixin,
    preprocess_image,
    retrieve_timesteps,
    remove_floater,
    remove_degenerate_face,
    reduce_face,
    smart_load_model,
    parts_augmentations
)
from transformers import (
    BitImageProcessor,
)

import shapeup_geometry
from shapeup_geometry.models.autoencoders.surface_extractors import MeshExtractResult
from shapeup_geometry.utils.config import ExperimentConfig, load_config
from omegaconf import DictConfig, OmegaConf
from ..autoencoders.michelangelo_autoencoder import MichelangeloAutoencoder
from ..conditional_encoders.dinov2_encoder import Dinov2Encoder
from ..conditional_encoders.t5_encoder import T5Encoder
from ..conditional_encoders.label_encoder import LabelEncoder
from ..transformers.flux_transformer_1d import FluxDenoiser
from shapeup_geometry.utils.saving import SaverMixin
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
from os.path import join as pjoin

class Step1X3DGeometryPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `torch.Tensor`):
            List of PIL images or a tensor representing the input images.
        meshes (`List[trimesh.Trimesh]` or `np.ndarray`)
            List of denoised trimesh meshes of length `batch_size` or a tuple of NumPy array with shape `((vertices, 3), (faces, 3)) of length `batch_size``.
    """

    image: PIL.Image.Image
    mesh: Union[trimesh.Trimesh, MeshExtractResult, np.ndarray]

# Where the released ShapeUp weights live on the Hub. The repo is laid out by
# modality so the texture checkpoint can sit alongside the geometry adapter:
#
#   Inbar2344/ShapeUP
#   ├── geometry/adapter-1024-550k/{adapter_config.json,adapter_model.safetensors}
#   └── texture/...
SHAPEUP_HUB_REPO = "Inbar2344/ShapeUP"
DEFAULT_GEOMETRY_ADAPTER_SUBFOLDER = "geometry/adapter-1024-550k"


class ShapeUpPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        meshes (`List[trimesh.Trimesh]` or `np.ndarray`)
            List of denoised trimesh meshes of length `batch_size` or a tuple of NumPy array with shape `((vertices, 3), (faces, 3)) of length `batch_size``.
    """
    mesh: Union[trimesh.Trimesh, MeshExtractResult, np.ndarray]


class Step1X3DGeometryPipeline(
    DiffusionPipeline, FromSingleFileMixin, TransformerDiffusionMixin
):
    """
    Step1X-3D Geometry Pipeline, generate high-quality meshes conditioned on image/caption/label inputs


    Args:
        scheduler (FlowMatchEulerDiscreteScheduler):
            The diffusion scheduler controlling the denoising process
        vae (MichelangeloAutoencoder):
            Variational Autoencoder for latent space compression/reconstruction
        transformer (FluxDenoiser):
            Transformer-based denoising model
        visual_encoder (Dinov2Encoder):
            Pretrained visual encoder for image feature extraction
        caption_encoder (T5Encoder):
            Text encoder for processing natural language captions
        label_encoder (LabelEncoder):
            Auxiliary text encoder for label conditioning
        visual_eature_extractor (BitImageProcessor):
            Preprocessor for input images

    Note:
        - CPU offloading sequence: visual_encoder → caption_encoder → label_encoder → transformer → vae
        - Optional components: visual_encoder, visual_eature_extractor, caption_encoder, label_encoder
    """

    model_cpu_offload_seq = (
        "visual_encoder->caption_encoder->label_encoder->transformer->vae"
    )
    _optional_components = [
        "visual_encoder",
        "visual_eature_extractor",
        "caption_encoder",
        "label_encoder",
    ]

    @classmethod
    def from_pretrained(cls, model_path, subfolder='.', **kwargs):
        local_model_path = smart_load_model(model_path, subfolder)
        return super().from_pretrained(local_model_path, **kwargs)

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: MichelangeloAutoencoder,
        transformer: FluxDenoiser,
        visual_encoder: Dinov2Encoder,
        caption_encoder: T5Encoder,
        label_encoder: LabelEncoder,
        visual_eature_extractor: BitImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            visual_encoder=visual_encoder,
            caption_encoder=caption_encoder,
            label_encoder=label_encoder,
            visual_eature_extractor=visual_eature_extractor,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def check_inputs(
        self,
        image,
    ):
        r"""
        Check if the inputs are valid. Raise an error if not.
        """
        if isinstance(image, str):
            assert os.path.isfile(image) or image.startswith(
                "http"
            ), "Input image must be a valid URL or a file path."
        elif isinstance(image, (torch.Tensor, PIL.Image.Image)):
            raise ValueError(
                "Input image must be a `torch.Tensor` or `PIL.Image.Image`."
            )

    def encode_image(self, image, device, num_meshes_per_prompt):
        dtype = next(self.visual_encoder.parameters()).dtype

        image_embeds = self.visual_encoder.encode_image(image)
        image_embeds = image_embeds.repeat_interleave(num_meshes_per_prompt, dim=0)

        uncond_image_embeds = self.visual_encoder.empty_image_embeds.repeat(
            image_embeds.shape[0], 1, 1
        ).to(image_embeds)

        return image_embeds, uncond_image_embeds

    def encode_caption(self, caption, device, num_meshes_per_prompt):
        dtype = next(self.label_encoder.parameters()).dtype

        caption_embeds = self.caption_encoder.encode_text([caption])
        caption_embeds = caption_embeds.repeat_interleave(num_meshes_per_prompt, dim=0)

        uncond_caption_embeds = self.caption_encoder.empty_text_embeds.repeat(
            caption_embeds.shape[0], 1, 1
        ).to(caption_embeds)

        return caption_embeds, uncond_caption_embeds

    def encode_label(self, label, device, num_meshes_per_prompt):
        dtype = next(self.label_encoder.parameters()).dtype

        label_embeds = self.label_encoder.encode_label([label])
        label_embeds = label_embeds.repeat_interleave(num_meshes_per_prompt, dim=0)

        uncond_label_embeds = self.label_encoder.empty_label_embeds.repeat(
            label_embeds.shape[0], 1, 1
        ).to(label_embeds)

        return label_embeds, uncond_label_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, str],
        label: Optional[str] = None,
        caption: Optional[str] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        num_meshes_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        generator: Optional[int] = None,
        latents: Optional[torch.FloatTensor] = None,
        force_remove_background: bool = False,
        background_color: List[int] = [255, 255, 255],
        foreground_ratio: float = 0.95,
        surface_extractor_type: Optional[str] = None,
        bounds: float = 1.05,
        mc_level: float = 0.0,
        octree_resolution: int = 384,
        output_type: str = "trimesh",
        do_remove_floater: bool = True,
        do_remove_degenerate_face: bool = False,
        do_reduce_face: bool = True,
        do_shade_smooth: bool = True,
        max_facenum: int = 200000,
        return_dict: bool = True,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.FloatTensor` or `PIL.Image.Image` or `str`):
                `Image`, or tensor representing an image batch, or path to an image file. The image will be encoded to
                its CLIP/DINO-v2 embedding which the DiT will be conditioned on.
            label (`str`):
                The label of the generated mesh, like {"symmetry": "asymmetry", "edge_type": "smooth"}
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps. More denoising steps usually lead to a higher quality mesh at the expense
                of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not provided, will use equally spaced timesteps.
            num_meshes_per_prompt (`int`, *optional*, defaults to 1):
                The number of meshes to generate per input image.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                Higher guidance scale encourages generation that closely matches the input image.
            generator (`int`, *optional*):
                A seed to make the generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents to use as inputs for mesh generation.
            force_remove_background (`bool`, *optional*, defaults to `False`):
                Whether to force remove the background from the input image before processing.
            background_color (`List[int]`, *optional*, defaults to `[255, 255, 255]`):
                RGB color values for the background if it needs to be removed or modified.
            foreground_ratio (`float`, *optional*, defaults to 0.95):
                Ratio of the image to consider as foreground when processing.
            surface_extractor_type (`str`, *optional*, defaults to "mc"):
                Type of surface extraction method to use ("mc" for Marching Cubes or other available methods).
            bounds (`float`, *optional*, defaults to 1.05):
                Bounding box size for the generated mesh.
            mc_level (`float`, *optional*, defaults to 0.0):
                Iso-surface level value for Marching Cubes extraction.
            octree_resolution (`int`, *optional*, defaults to 256):
                Resolution of the octree used for mesh generation.
            output_type (`str`, *optional*, defaults to "trimesh"):
                Type of output mesh format ("trimesh" or other supported formats).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `MeshPipelineOutput` instead of a plain tuple.

        Returns:
            [`MeshPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`MeshPipelineOutput`] is returned, otherwise a `tuple` is returned where the
                first element is a list of generated meshes and the second element is a list of corresponding metadata.
        """
        # 0. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
        )
        device = self._execution_device
        self._guidance_scale = guidance_scale

        # 1. Define call parameters
        if isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        elif isinstance(image, PIL.Image.Image) or isinstance(image, str):
            batch_size = 1

        # 2. Preprocess input image
        if isinstance(image, torch.Tensor):
            assert image.ndim == 3  # H, W, 3
            image_pil = TF.to_pil_image(image)
        elif isinstance(image, PIL.Image.Image):
            image_pil = image
        elif isinstance(image, str):
            if image.startswith("http"):
                import requests

                image_pil = PIL.Image.open(requests.get(image, stream=True).raw)
            else:
                image_pil = PIL.Image.open(image)
        image_pil = preprocess_image(image_pil, force=force_remove_background, background_color=background_color, foreground_ratio=foreground_ratio)  # remove the background images

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image_pil, device, num_meshes_per_prompt
        )
        if self.do_classifier_free_guidance and image_embeds is not None:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)
        # 3.1 Encode label condition
        label_embeds = None
        if self.transformer.cfg.use_label_condition:
            if label is not None:
                label_embeds, negative_label_embeds = self.encode_label(
                    label, device, num_meshes_per_prompt
                )
                if self.do_classifier_free_guidance:
                    label_embeds = torch.cat(
                        [negative_label_embeds, label_embeds], dim=0
                    )
            else:
                uncond_label_embeds = self.label_encoder.empty_label_embeds.repeat(
                    num_meshes_per_prompt, 1, 1
                ).to(image_embeds)
                if self.do_classifier_free_guidance:
                    label_embeds = torch.cat(
                        [uncond_label_embeds, uncond_label_embeds], dim=0
                    )
        # 3.3 Encode caption condition
        caption_embeds = None
        if self.transformer.cfg.use_caption_condition:
            if caption is not None:
                caption_embeds, negative_caption_embeds = self.encode_caption(
                    caption, device, num_meshes_per_prompt
                )
                if self.do_classifier_free_guidance:
                    caption_embeds = torch.cat(
                        [negative_caption_embeds, caption_embeds], dim=0
                    )
            else:
                uncond_caption_embeds = self.caption_encoder.empty_text_embeds.repeat(
                    num_meshes_per_prompt, 1, 1
                ).to(image_embeds)
                if self.do_classifier_free_guidance:
                    caption_embeds = torch.cat(
                        [uncond_caption_embeds, uncond_caption_embeds], dim=0
                    )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_latents = self.vae.cfg.num_latents
        num_channels_latents = self.transformer.cfg.input_channels
        latents = self.prepare_latents(
            batch_size * num_meshes_per_prompt,
            num_latents,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    visual_condition=image_embeds,
                    label_condition=label_embeds,
                    caption_condition=caption_embeds,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                if (i <= zero_steps) and use_zero_init:
                    noise_pred = noise_pred * 0.0

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 4. Post-processing
        if not output_type == "latent":
            if latents.dtype == torch.bfloat16:
                self.vae.to(torch.float16)
                latents = latents.to(torch.float16)
            mesh = self.vae.extract_geometry(
                self.vae.decode(latents),
                surface_extractor_type=surface_extractor_type,
                bounds=bounds,
                mc_level=mc_level,
                octree_resolution=octree_resolution,
                enable_pbar=False,
            )
            if output_type != "raw":
                mesh_list = []
                for i, cur_mesh in enumerate(mesh):
                    print(f"Generating mesh {i+1}/{num_meshes_per_prompt}")
                    if output_type == "trimesh":
                        import trimesh

                        cur_mesh = trimesh.Trimesh(
                            vertices=cur_mesh.verts.cpu().numpy(),
                            faces=cur_mesh.faces.cpu().numpy(),
                        )
                        cur_mesh.fix_normals()
                        cur_mesh.face_normals
                        cur_mesh.vertex_normals
                        cur_mesh.visual = trimesh.visual.TextureVisuals(
                            material=trimesh.visual.material.PBRMaterial(
                                baseColorFactor=(255, 255, 255),
                                main_color=(255, 255, 255),
                                metallicFactor=0.05,
                                roughnessFactor=1.0,
                            )
                        )
                        if do_remove_floater:
                            cur_mesh = remove_floater(cur_mesh)
                        if do_remove_degenerate_face:
                            cur_mesh = remove_degenerate_face(cur_mesh)
                        if do_reduce_face and max_facenum > 0:
                            cur_mesh = reduce_face(cur_mesh, max_facenum)
                        if do_shade_smooth:
                            cur_mesh = cur_mesh.smooth_shaded
                        mesh_list.append(cur_mesh)
                    elif output_type == "np":
                        if do_remove_floater:
                            print(
                                'remove floater is NOT used when output_type is "np". '
                            )
                        if do_remove_degenerate_face:
                            print(
                                'remove degenerate face is NOT used when output_type is "np". '
                            )
                        if do_reduce_face:
                            print(
                                'reduce floater is NOT used when output_type is "np". '
                            )
                        if do_shade_smooth:
                            print('shade smooth is NOT used when output_type is "np". ')
                        mesh_list.append(
                            [
                                cur_mesh[0].verts.cpu().numpy(),
                                cur_mesh[0].faces.cpu().numpy(),
                            ]
                        )
                mesh = mesh_list
            else:
                if do_remove_floater:
                    print('remove floater is NOT used when output_type is "raw". ')
                if do_remove_degenerate_face:
                    print(
                        'remove degenerate face is NOT used when output_type is "raw". '
                    )
                if do_reduce_face:
                    print('reduce floater is NOT used when output_type is "raw". ')

        else:
            mesh = latents

        if not return_dict:
            return tuple(image_pil), tuple(mesh)
        return Step1X3DGeometryPipelineOutput(image=image_pil, mesh=mesh)

class VAEPipeline(SaverMixin):
    """
    Step1X-3D GeomeVAEtry Pipeline, generate high-quality meshes conditioned on image/caption/label inputs
    """
    def __init__(
        self,
        config_path,
        float_precision='high',
        **kwargs,
    ):
        
        n_gpus = kwargs.get('n_gpus', 1)
        cli_args = kwargs.get('cli_args', [])
        torch.set_float32_matmul_precision(float_precision)
        cfg: ExperimentConfig
        self.cfg = load_config(config_path, cli_args=cli_args, n_gpus=n_gpus)
        self.shape_model = shapeup_geometry.find(self.cfg.system.shape_model_type)(
            self.cfg.system.shape_model
        )


    @torch.no_grad()
    def __call__(self, data_dir=None, data_type=None):
        self.set_save_dir(os.path.join(self.cfg.trial_dir, "save"))
        self.shape_model.eval()
        if data_type is not None:
            self.cfg.data_type = data_type
        if data_dir is not None:
            self.cfg.data.root_dir = data_dir
        data_module = shapeup_geometry.find(self.cfg.data_type)(self.cfg.data)
        data_module.setup('test')
        data_loader = data_module.test_dataloader()
        num_point_feats = 3 + self.cfg.system.shape_model.point_feats
        for batch in tqdm(data_loader):
            start_time = time.time()
            shape_latents, kl_embed, posterior = self.shape_model.encode(
            batch["surface"][..., :num_point_feats],
            sharp_surface=(
                batch["sharp_surface"][..., :num_point_feats]
                if "sharp_surface" in batch
                else None
            ),
            sample_posterior=self.cfg.system.sample_posterior,
            )
            end_time = time.time()
            print(f'encodeing start time: {start_time}, encoding end time {end_time}')
            latents = self.shape_model.decode(kl_embed) # [B, num_latents, width]
            meshes = self.shape_model.extract_geometry(
                latents,
                bounds=self.cfg.system.bounds,
                mc_level=self.cfg.system.mc_level,
                octree_resolution=self.cfg.system.octree_resolution,
                enable_pbar=False,
            )
            for idx, name in enumerate(batch["uid"]):
                self.save_mesh(
                    f"{name}.obj",
                    meshes[idx].verts,
                    meshes[idx].faces,
                )
            
            torch.cuda.empty_cache()

class VAEDataCreationPipeline(SaverMixin):
    """
    Step1X-3D GeomeVAEtry Pipeline, generate high-quality meshes conditioned on image/caption/label inputs
    """
    def __init__(
        self,
        config_path,
        float_precision='high',
        **kwargs,
    ):
        
        n_gpus = kwargs.get('n_gpus', 1)
        cli_args = kwargs.get('cli_args', [])
        torch.set_float32_matmul_precision(float_precision)
        cfg: ExperimentConfig
        cfg = load_config(config_path, cli_args=cli_args, n_gpus=n_gpus)
        self.cfg = cfg
        self.shape_model = shapeup_geometry.find(self.cfg.system.shape_model_type)(
            self.cfg.system.shape_model
        )


    @torch.no_grad()
    def __call__(self, data_dir=None, data_type=None, ids_list=None):
        self.shape_model.eval()
        if data_type is not None:
            self.cfg.data_type = data_type
        if data_dir is not None:
            self.cfg.data.root_dir = data_dir
        surfaces_dir = pjoin(self.cfg.data.root_dir, 'surfaces')
        if ids_list is None:
            ids_list = sorted(
                d for d in os.listdir(surfaces_dir)
                if os.path.isdir(pjoin(surfaces_dir, d))
            )

        # keep only uids the loader can actually read: at least one part/frame
        # subdirectory, and every such subdirectory must contain a pc.npz.
        def _is_processable(uid):
            obj_dir = pjoin(surfaces_dir, uid)
            if not os.path.isdir(obj_dir):
                return False
            sub_dirs = [d for d in os.listdir(obj_dir) if os.path.isdir(pjoin(obj_dir, d))]
            return len(sub_dirs) > 0 and all(
                os.path.exists(pjoin(obj_dir, sd, "pc.npz")) for sd in sub_dirs
            )

        processable = [uid for uid in ids_list if _is_processable(uid)]
        skipped = [uid for uid in ids_list if uid not in processable]
        if skipped:
            print(f"Skipping {len(skipped)} surfaces without complete pc.npz data: {skipped}")

        self.cfg.data.ids_list = processable
        print(f"Processing {len(processable)} surfaces from {surfaces_dir}")
        data_module = shapeup_geometry.find(self.cfg.data_type)(self.cfg.data)
        data_module.setup('test')
        data_loader = data_module.test_dataloader()
        num_point_feats = 3 + self.cfg.system.shape_model.point_feats
        
        for batch in tqdm(data_loader):
            obj_dir = pjoin(self.cfg.data.root_dir, 'surfaces', batch['uid'][0])
            self.set_save_dir(obj_dir)
            npz_path = pjoin(obj_dir, "pc.npz")
            print(f"Processing file {batch['uid'][0]}")
            n_parts, N = batch["objs_surface"][0].shape[:2]
            objs_surface = batch["objs_surface"][0]
            objs_sharp_surface = batch["objs_sharp_surface"][0]

            shape_latents, kl_embed, posterior = self.shape_model.encode(
            objs_surface[..., :num_point_feats],
            sharp_surface=objs_sharp_surface,
            sample_posterior=self.cfg.system.sample_posterior,
            )
            ### save_new_npz file 
            posterior_mean = posterior.mean.detach().cpu().numpy()
            posterior_std = posterior.std.detach().cpu().numpy()
            data_dict = {key: batch[key] for key in batch}
            data_dict['objs_posterior_mean'] = posterior_mean
            data_dict['objs_posterior_std'] = posterior_std
            np.savez(npz_path, **data_dict)
            torch.cuda.empty_cache()      

                        
class ShapeUpGeometryPipeline(
    DiffusionPipeline, FromSingleFileMixin, TransformerDiffusionMixin
):
    """
    ShapeUp geometry (shape-editing) inference pipeline.

    Given a source shape (its point cloud / surface) and a target edit image, it
    generates the edited, untextured mesh with a rectified-flow DiT conditioned on
    (a) the DINOv2+CLIP embedding of the edit image and (b) the VAE latents of the
    source shape.

    Weights come from their original sources — nothing is duplicated in the ShapeUp
    checkpoint:
      * ``vae`` (MichelangeloAutoencoder) and the DiT base weights + visual/label
        projections load from the Step1X-3D weights on the Hub (in each module's
        ``configure()``);
      * ``visual_encoder`` (DINOv2+CLIP) loads from the Hub;
      * only the trained **adapter** (LoRA on the DiT + ``proj_shape_condtion``) is
        loaded from the slimmed ShapeUp checkpoint produced by
        ``extract_geometry_adapter.py``.

    Unlike training — where the source-shape VAE posterior is precomputed — the
    source shape is encoded through the VAE at inference time (see ``__call__``),
    mirroring ``ShapeupRectifiedFlowSystem.test_step``/``sample``.

    Build with :meth:`from_config`; the raw ``__init__`` just registers the modules.
    """

    model_cpu_offload_seq = "visual_encoder->transformer->vae"
    _optional_components = ["visual_encoder"]

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        config_path: str,
        adapter_path: Optional[str] = None,
        adapter_subfolder: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
        cli_args: Optional[List[str]] = None,
    ):
        """Build the pipeline from a ShapeUp geometry-diffusion yaml config.

        Args:
            config_path: path to the geometry-diffusion inference config.
            adapter_path: where to get the trained adapter (``adapter_config.json`` +
                ``adapter_model.safetensors``, see ``extract_geometry_adapter.py``).
                Either a local directory, or a Hub repo id such as
                ``"Inbar2344/ShapeUP"`` -- anything that is not an existing directory
                is treated as a repo id and downloaded.
                If ``None``, no adapter is loaded (base Step1X-3D behaviour).
            adapter_subfolder: subfolder within the Hub repo; defaults to
                ``"geometry/adapter-1024-550k"``. Ignored for a local directory
                unless the adapter files live in a subfolder of it.
            device / dtype: where/what to place the modules in.
            cli_args: optional OmegaConf-style overrides for the config.
        """
        cfg = load_config(config_path, cli_args=cli_args or [], n_gpus=1)
        return cls.from_spec(cfg.system, adapter_path=adapter_path,
                             adapter_subfolder=adapter_subfolder,
                             device=device, dtype=dtype)

    @classmethod
    def from_spec(
        cls,
        sys_cfg,
        adapter_path: Optional[str] = None,
        adapter_subfolder: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """Build the pipeline from an in-memory ``system`` spec.

        Same as :meth:`from_config` but takes the ``system`` block directly (a dict
        or OmegaConf node) instead of reading a training yaml, so an inference
        script can pin every architecture and sampling value itself and stay
        independent of the training configs.
        """
        if not isinstance(sys_cfg, DictConfig):
            sys_cfg = OmegaConf.create(sys_cfg)

        # vae + denoiser + visual encoder each load their base weights from the Hub
        vae = shapeup_geometry.find(sys_cfg.shape_model_type)(sys_cfg.shape_model)
        transformer = shapeup_geometry.find(sys_cfg.denoiser_model_type)(
            sys_cfg.denoiser_model
        )
        visual_encoder = shapeup_geometry.find(sys_cfg.visual_condition_type)(
            sys_cfg.visual_condition
        )
        scheduler = shapeup_geometry.find(sys_cfg.denoise_scheduler_type)(
            **sys_cfg.denoise_scheduler
        )

        # attach + load the trained adapter (LoRA on the DiT + shape projection)
        if sys_cfg.get("use_lora", False) and adapter_path is not None:
            cls._load_adapter(transformer, adapter_path, adapter_subfolder)

        pipeline = cls(
            scheduler=scheduler,
            vae=vae,
            transformer=transformer,
            visual_encoder=visual_encoder,
            n_part_latents=sys_cfg.get("n_part_latents", 1024),
            cfg_type=sys_cfg.get("cfg_type", "single"),
            guidance_scale=sys_cfg.get("guidance_scale", 7.5),
            guidance_scale_visual=sys_cfg.get("guidance_scale_visual", 2.5),
            guidance_scale_shape=sys_cfg.get("guidance_scale_shape", 2.5),
            num_inference_steps=sys_cfg.get("num_inference_steps", 30),
            bounds=sys_cfg.get("bounds", 1.05),
            mc_level=sys_cfg.get("mc_level", 0.0),
            octree_resolution=sys_cfg.get("octree_resolution", 256),
        )
        pipeline.vae.eval()
        pipeline.transformer.eval()
        pipeline.visual_encoder.eval()
        pipeline.to(device, dtype)
        return pipeline

    @staticmethod
    def _resolve_adapter_path(adapter_path, adapter_subfolder=None):
        """Turn a local directory or a Hub repo id into a local directory.

        A path that already exists on disk is used as-is (optionally descending into
        ``adapter_subfolder``); anything else is treated as a Hub repo id and fetched,
        defaulting to the geometry adapter's subfolder in the ShapeUp release repo.
        """
        if os.path.isdir(adapter_path):
            if adapter_subfolder:
                local = pjoin(adapter_path, adapter_subfolder)
                if not os.path.isdir(local):
                    raise FileNotFoundError(
                        f"No subfolder '{adapter_subfolder}' under '{adapter_path}'"
                    )
                return local
            return adapter_path

        subfolder = adapter_subfolder or DEFAULT_GEOMETRY_ADAPTER_SUBFOLDER
        print(f"Fetching adapter '{subfolder}' from Hub repo '{adapter_path}'")
        return smart_load_model(adapter_path, subfolder)

    @staticmethod
    def _load_adapter(transformer, adapter_path, adapter_subfolder=None):
        """Attach the LoRA adapter to the DiT and load LoRA + proj_shape weights."""
        adapter_path = ShapeUpGeometryPipeline._resolve_adapter_path(
            adapter_path, adapter_subfolder
        )
        with open(pjoin(adapter_path, "adapter_config.json"), "r") as f:
            acfg = json.load(f)
        transformer.dit_model.add_adapter(
            LoraConfig(
                r=acfg["rank"],
                lora_alpha=acfg["alpha"],
                init_lora_weights=acfg.get("init_lora_weights", "gaussian"),
                target_modules=acfg["target_modules"],
            )
        )
        adapter_sd = load_file(pjoin(adapter_path, "adapter_model.safetensors"))

        # The manifest (written by extract_geometry_adapter.py) records how much
        # trainable state the checkpoint held, so a truncated or stale adapter file is
        # caught here rather than silently producing a half-trained model.
        n_elements = sum(v.numel() for v in adapter_sd.values())
        for field, actual in (
            ("num_tensors", len(adapter_sd)),
            ("num_elements", n_elements),
        ):
            expected = acfg.get(field)
            if expected is not None and expected != actual:
                raise RuntimeError(
                    f"Adapter {field} mismatch: manifest says {expected}, "
                    f"file has {actual} ({adapter_path})"
                )

        missing, unexpected = transformer.load_state_dict(adapter_sd, strict=False)
        # `unexpected` means an adapter tensor has no home in the model, i.e. it was
        # NOT loaded -- never accept that, it is trained state being dropped.
        if unexpected:
            raise RuntimeError(f"Unexpected adapter keys: {unexpected[:5]} ...")
        # The reverse direction: a LoRA / shape-projection parameter that the model has
        # but the adapter did not supply would still be at its random init.
        still_missing = [
            k for k in missing if ".lora_" in k or "proj_shape_condtion" in k
        ]
        if still_missing:
            raise RuntimeError(
                f"Adapter keys not loaded: {still_missing[:5]} ..."
            )
        print(
            f"Loaded geometry adapter ({len(adapter_sd)} tensors, "
            f"{n_elements/1e6:.1f}M params) from {adapter_path}"
        )

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: MichelangeloAutoencoder,
        transformer: FluxDenoiser,
        visual_encoder,
        n_part_latents: int = 1024,
        cfg_type: str = "single",
        guidance_scale: float = 7.5,
        guidance_scale_visual: float = 2.5,
        guidance_scale_shape: float = 2.5,
        num_inference_steps: int = 30,
        bounds: float = 1.05,
        mc_level: float = 0.0,
        octree_resolution: int = 256,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            visual_encoder=visual_encoder,
        )
        # inference knobs carried over from the training/inference config
        self.n_part_latents = n_part_latents
        self.cfg_type = cfg_type
        self.default_guidance_scale = guidance_scale
        self.default_guidance_scale_visual = guidance_scale_visual
        self.default_guidance_scale_shape = guidance_scale_shape
        self.default_num_inference_steps = num_inference_steps
        self.default_bounds = bounds
        self.default_mc_level = mc_level
        self.default_octree_resolution = octree_resolution

    # ------------------------------------------------------------------ #
    #  Encoding helpers
    # ------------------------------------------------------------------ #
    def _prepare_images(self, images, device):
        """Return an image tensor [B, H, W, 3] in [0, 1] from tensor / paths / PIL."""
        if torch.is_tensor(images):
            return images.to(device)
        if isinstance(images, (str, PIL.Image.Image)):
            images = [images]
        pil_images = [
            PIL.Image.open(im) if isinstance(im, str) else im for im in images
        ]
        # crop / center / composite on a white background (RGBA), then drop alpha:
        # the encoder's CLIP/DINO transforms expect RGB [B, H, W, 3] in [0, 1].
        pil_images = preprocess_image(pil_images)
        tensors = [
            torch.from_numpy(np.array(im.convert("RGB")) / 255.0).float()
            for im in pil_images
        ]
        return torch.stack(tensors, dim=0).to(device)

    def encode_shape(self, batch, device):
        """Encode the source surface through the VAE and select the shape condition.

        Mirrors ``ShapeupRectifiedFlowSystem.test_step``: encode -> KL latents ->
        gather ``n_part_latents`` random tokens as the shape condition.
        """
        vae_dtype = next(self.vae.parameters()).dtype
        surface = batch["surface"][..., :6].to(device=device, dtype=vae_dtype)
        sharp_surface = batch.get("sharp_surface", None)
        if sharp_surface is not None:
            sharp_surface = sharp_surface[..., :6].to(device=device, dtype=vae_dtype)

        _, kl_embed, _ = self.vae.encode(
            surface, sample_posterior=True, sharp_surface=sharp_surface
        )
        bs, n_latents, latent_dim = kl_embed.shape
        rand_scores = torch.rand(bs, n_latents, device=kl_embed.device)
        _, indices = torch.sort(rand_scores, dim=1)
        indices = indices[:, : self.n_part_latents]
        indices = indices.unsqueeze(-1).expand(-1, -1, latent_dim)
        shape_cond = torch.gather(kl_embed, dim=1, index=indices)
        return shape_cond.contiguous()

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(
        self,
        batch: Dict[str, Any],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_scale_visual: Optional[float] = None,
        guidance_scale_shape: Optional[float] = None,
        cfg_type: Optional[str] = None,
        seed: Optional[int] = None,
        eta: float = 0.0,
        surface_extractor_type: Optional[str] = None,
        bounds: Optional[float] = None,
        mc_level: Optional[float] = None,
        octree_resolution: Optional[int] = None,
        output_type: str = "trimesh",
        do_remove_floater: bool = True,
        do_remove_degenerate_face: bool = False,
        do_reduce_face: bool = True,
        do_shade_smooth: bool = True,
        max_facenum: int = 200000,
        match_reference: bool = False,
        return_dict: bool = True,
    ):
        r"""Generate the edited mesh(es).

        Args:
            batch: dict with
                ``"surface"``  (`[B, N, 6]`) source shape point cloud (xyz+normal),
                ``"sharp_surface"`` (`[B, N, 6]`, optional) sharp-edge samples,
                ``"image"``    edit image(s): a `[B, H, W, 3]` tensor in `[0, 1]`,
                               or a list of file paths / PIL images.
            match_reference: reproduce ``ShapeupRectifiedFlowSystem.test_step``
                bit-for-bit instead of using this pipeline's own conventions. That
                code seeds the global RNG and passes *no* generator to the sampler,
                so its initial latent noise continues the global CUDA stream after
                the VAE posterior sample and the token-selection ``torch.rand``; a
                private generator is a different draw. It also decodes in the
                model's own dtype (DeepSpeed ran the whole model in bf16). Combine
                with ``dtype=torch.bfloat16`` in :meth:`from_config` and with the
                post-processing flags off to compare against a saved ``w_cfg.obj``,
                which is raw marching-cubes output.
            The remaining args default to the values baked in from the config.
        """
        from shapeup_geometry.systems.utils import (
            flow_sample,
            flow_sample_separate_cfg,
        )

        device = self._execution_device
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        cfg_type = cfg_type or self.cfg_type
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.default_guidance_scale
        )
        guidance_scale_visual = (
            guidance_scale_visual
            if guidance_scale_visual is not None
            else self.default_guidance_scale_visual
        )
        guidance_scale_shape = (
            guidance_scale_shape
            if guidance_scale_shape is not None
            else self.default_guidance_scale_shape
        )
        bounds = bounds if bounds is not None else self.default_bounds
        mc_level = mc_level if mc_level is not None else self.default_mc_level
        octree_resolution = (
            octree_resolution
            if octree_resolution is not None
            else self.default_octree_resolution
        )

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # test_step passes no generator, so its noise comes from the global stream
            # *after* encode_shape's draws -- keep that stream to match it.
            generator = (
                None
                if match_reference
                else torch.Generator(device=device).manual_seed(seed)
            )
        else:
            generator = None

        # 1. source-shape condition (VAE encode + token selection)
        shape_cond = self.encode_shape(batch, device)

        # 2. visual (edit image) condition
        enc_dtype = next(self.visual_encoder.parameters()).dtype
        images = self._prepare_images(batch["image"], device).to(enc_dtype)
        visual_cond = self.visual_encoder.encode_image(images)
        visual_cond = visual_cond.to(shape_cond.dtype)
        empty_visual = self.visual_encoder.empty_image_embeds.repeat(
            visual_cond.shape[0], 1, 1
        ).to(visual_cond)

        # 3. classifier-free-guidance conditions + rectified-flow sampling
        do_cfg = cfg_type == "single" and guidance_scale != 1.0
        do_cfg_sep = cfg_type != "single" and (
            guidance_scale_visual != 1.0 or guidance_scale_shape != 1.0
        )

        if cfg_type == "single":
            if do_cfg:
                vis = torch.cat([empty_visual, visual_cond], dim=0)
                shp = torch.cat([torch.zeros_like(shape_cond), shape_cond], dim=0)
            else:
                vis, shp = visual_cond, shape_cond
            sample_loop = flow_sample(
                self.scheduler,
                self.transformer.eval(),
                shape=self.vae.latent_shape,
                visual_cond=vis,
                caption_cond=None,
                label_cond=None,
                shape_cond=shp,
                steps=num_inference_steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_cfg,
                device=device,
                eta=eta,
                disable_prog=False,
                generator=generator,
            )
        else:  # "separate_visual" / "separate"
            if do_cfg_sep:
                if cfg_type == "separate":
                    vis = torch.cat([empty_visual, empty_visual, visual_cond], dim=0)
                    shp = torch.cat(
                        [torch.zeros_like(shape_cond), shape_cond, shape_cond], dim=0
                    )
                else:  # separate_visual
                    vis = torch.cat([empty_visual, visual_cond, visual_cond], dim=0)
                    shp = torch.cat(
                        [
                            torch.zeros_like(shape_cond),
                            torch.zeros_like(shape_cond),
                            shape_cond,
                        ],
                        dim=0,
                    )
            else:
                vis, shp = visual_cond, shape_cond
            sample_loop = flow_sample_separate_cfg(
                self.scheduler,
                self.transformer.eval(),
                shape=self.vae.latent_shape,
                visual_cond=vis,
                caption_cond=None,
                label_cond=None,
                shape_cond=shp,
                steps=num_inference_steps,
                guidance_scale_visual=guidance_scale_visual,
                guidance_scale_shape=guidance_scale_shape,
                do_classifier_free_guidance=do_cfg_sep,
                device=device,
                eta=eta,
                disable_prog=False,
                generator=generator,
            )

        latents = None
        for sample, t in sample_loop:
            latents = sample

        # 4. decode latents -> geometry -> mesh
        if output_type == "latent":
            mesh = latents
            if not return_dict:
                return (mesh,)
            return ShapeUpPipelineOutput(mesh=mesh)

        # The fp16 down-cast rescues a bf16 latent meeting an fp32 VAE; test_step had
        # no such mismatch (whole model in bf16) and decoded in bf16 directly.
        if latents.dtype == torch.bfloat16 and not match_reference:
            self.vae.to(torch.float16)
            latents = latents.to(torch.float16)
        geometry_latents = self.vae.decode(latents)
        mesh = self.vae.extract_geometry(
            geometry_latents,
            surface_extractor_type=surface_extractor_type,
            bounds=bounds,
            mc_level=mc_level,
            octree_resolution=octree_resolution,
            enable_pbar=False,
        )

        if output_type == "raw":
            if not return_dict:
                return tuple(mesh)
            return ShapeUpPipelineOutput(mesh=mesh)

        mesh_list = []
        for i, cur_mesh in enumerate(mesh):
            if (
                cur_mesh.verts is None
                or cur_mesh.verts.shape[0] == 0
                or cur_mesh.faces is None
                or cur_mesh.faces.shape[0] == 0
            ):
                mesh_list.append(None)
                continue
            if output_type == "trimesh":
                import trimesh

                cur_mesh = trimesh.Trimesh(
                    vertices=cur_mesh.verts.cpu().numpy(),
                    faces=cur_mesh.faces.cpu().numpy(),
                )
                cur_mesh.fix_normals()
                cur_mesh.face_normals
                cur_mesh.vertex_normals
                cur_mesh.visual = trimesh.visual.TextureVisuals(
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorFactor=(255, 255, 255),
                        main_color=(255, 255, 255),
                        metallicFactor=0.05,
                        roughnessFactor=1.0,
                    )
                )
                if do_remove_floater:
                    cur_mesh = remove_floater(cur_mesh)
                if do_remove_degenerate_face:
                    cur_mesh = remove_degenerate_face(cur_mesh)
                if do_reduce_face and max_facenum > 0:
                    cur_mesh = reduce_face(cur_mesh, max_facenum)
                if do_shade_smooth:
                    cur_mesh = cur_mesh.smooth_shaded
                mesh_list.append(cur_mesh)
            elif output_type == "np":
                mesh_list.append(
                    [cur_mesh.verts.cpu().numpy(), cur_mesh.faces.cpu().numpy()]
                )
        mesh = mesh_list

        if not return_dict:
            return tuple(mesh)
        return ShapeUpPipelineOutput(mesh=mesh)
