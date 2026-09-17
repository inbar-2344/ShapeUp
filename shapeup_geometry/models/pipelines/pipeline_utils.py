from typing import Callable, List, Optional, Union, Dict, Any
import os
from diffusers.utils import logging
import PIL.Image
import torch
import trimesh
import pymeshlab
import tempfile
from shapeup_geometry.models.autoencoders.surface_extractors import MeshExtractResult

logger = logging.get_logger(__name__)


def preprocess_image(
    images_pil: Union[List[PIL.Image.Image], PIL.Image.Image],
    force: bool = False,
    background_color: List[int] = [255, 255, 255],
    foreground_ratio: float = 0.9,
    rembg_backend: str = "bria",
    **rembg_kwargs,
):
    r"""
    Crop and remote the background of the input image
    Args:
        image_pil (`List[PIL.Image.Image]`):
            List of `PIL.Image.Image` objects representing the input image.
        force (`bool`, *optional*, defaults to `False`):
            Whether to force remove the background even if the image has an alpha channel.
    Returns:
        `List[PIL.Image.Image]`: List of `PIL.Image.Image` objects representing the preprocessed image.
    """
    is_single_image = False
    if isinstance(images_pil, PIL.Image.Image):
        images_pil = [images_pil]
        is_single_image = True
    preprocessed_images = []
    for i in range(len(images_pil)):
        image = images_pil[i]
        width, height, size = image.width, image.height, image.size
        do_remove = True
        if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
            # explain why current do not rm bg
            print(
                "alhpa channl not empty, skip remove background, using alpha channel as mask"
            )
            do_remove = False
        do_remove = do_remove or force
        if do_remove:
            import rembg  # lazy import

            if rembg_backend == "default":
                image = rembg.remove(image, **rembg_kwargs)
            else:
                image = rembg.remove(
                    image,
                    session=rembg.new_session(
                        model_name="bria",
                        providers=[
                            (
                                "CUDAExecutionProvider",
                                {
                                    "device_id": 0,
                                    "arena_extend_strategy": "kSameAsRequested",
                                    "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
                                    "cudnn_conv_algo_search": "HEURISTIC",
                                },
                            ),
                            "CPUExecutionProvider",
                        ],
                    ),
                    **rembg_kwargs,
                )

        # calculate the min bbox of the image
        alpha = image.split()[-1]
        bboxs = alpha.getbbox()
        x1, y1, x2, y2 = bboxs
        dy, dx = y2 - y1, x2 - x1
        s = min(height * foreground_ratio / dy, width * foreground_ratio / dx)
        Ht, Wt = int(dy * s), int(dx * s)

        background = PIL.Image.new("RGBA", image.size, (*background_color, 255))
        image = PIL.Image.alpha_composite(background, image)
        image = image.crop(alpha.getbbox())
        alpha = alpha.crop(alpha.getbbox())

        # Calculate the new size after rescaling
        new_size = tuple(int(dim * foreground_ratio) for dim in size)
        # Resize the image while maintaining the aspect ratio
        resized_image = image.resize((Wt, Ht))
        resized_alpha = alpha.resize((Wt, Ht))
        # Create a new image with the original size and white background
        padded_image = PIL.Image.new("RGB", size, tuple(background_color))
        padded_alpha = PIL.Image.new("L", size, (0))
        paste_position = (
            (width - resized_image.width) // 2,
            (height - resized_image.height) // 2,
        )
        padded_image.paste(resized_image, paste_position)
        padded_alpha.paste(resized_alpha, paste_position)

        # expand image to 1:1
        width, height = padded_image.size
        if width == height:
            padded_image.putalpha(padded_alpha)
            preprocessed_images.append(padded_image)
            continue
        new_size = (max(width, height), max(width, height))
        new_image = PIL.Image.new("RGB", new_size, tuple(background_color))
        new_alpha = PIL.Image.new("L", new_size, (0))
        paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
        new_image.paste(padded_image, paste_position)
        new_alpha.paste(padded_alpha, paste_position)
        new_image.putalpha(new_alpha)
        preprocessed_images.append(new_image)

    if is_single_image:
        return preprocessed_images[0]
    return preprocessed_images


def load_mesh(path):
    if path.endswith(".glb"):
        mesh = trimesh.load(path)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = trimesh.load(temp_file.name)
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def import_mesh(mesh):
    mesh_type = type(mesh)
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, MeshExtractResult):
        mesh = pymeshlab.MeshSet()
        mesh_pymeshlab = pymeshlab.Mesh(
            vertex_matrix=mesh.verts.cpu().numpy(), face_matrix=mesh.faces.cpu().numpy()
        )
        mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh, mesh_type


def remove_floater(mesh):
    mesh, mesh_type = import_mesh(mesh)

    mesh.apply_filter(
        "compute_selection_by_small_disconnected_components_per_face", nbfaceratio=0.001
    )
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")

    return pymeshlab2trimesh(mesh)


def remove_degenerate_face(mesh):
    mesh, mesh_type = import_mesh(mesh)

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)

    return pymeshlab2trimesh(mesh)


def reduce_face(mesh, max_facenum=50000):
    mesh, mesh_type = import_mesh(mesh)

    if max_facenum > mesh.current_mesh().face_number():
        return pymeshlab2trimesh(mesh)

    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True,
    )

    return pymeshlab2trimesh(mesh)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class TransformerDiffusionMixin:
    r"""
    Helper for DiffusionPipeline with vae and transformer.(mainly for DIT)
    """

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def fuse_qkv_projections(self, transformer: bool = True, vae: bool = True):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            transformer (`bool`, defaults to `True`): To apply fusion on the Transformer.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
        self.fusing_transformer = False
        self.fusing_vae = False

        if transformer:
            self.fusing_transformer = True
            self.transformer.fuse_qkv_projections()

        if vae:
            self.fusing_vae = True
            self.vae.fuse_qkv_projections()

    def unfuse_qkv_projections(self, transformer: bool = True, vae: bool = True):
        """Disable QKV projection fusion if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            transformer (`bool`, defaults to `True`): To apply fusion on the Transformer.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.

        """
        if transformer:
            if not self.fusing_transformer:
                logger.warning(
                    "The UNet was not initially fused for QKV projections. Doing nothing."
                )
            else:
                self.transformer.unfuse_qkv_projections()
                self.fusing_transformer = False

        if vae:
            if not self.fusing_vae:
                logger.warning(
                    "The VAE was not initially fused for QKV projections. Doing nothing."
                )
            else:
                self.vae.unfuse_qkv_projections()
                self.fusing_vae = False

def try_download(model_id, subfolder):
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=model_id,
            allow_patterns=[f"{subfolder}/*"],
        )
        print(path)
        model_path = os.path.join(path, subfolder)
        return model_path
    except Exception as e:
        raise e


def smart_load_model(model_path, subfolder = ""):
    if subfolder == "":
        if os.path.exists(model_path):
            return model_path
        else:
            return try_download(model_path, '.')
    else:
        if os.path.exists(os.path.join(model_path, subfolder)):
            return os.path.join(model_path, subfolder)
        else:
            return try_download(model_path, subfolder)


def rotation(pc: torch.Tensor) -> torch.Tensor:
    """
    Applies a random 3D rotation to each point cloud in the batch, without a for loop.
    
    Args:
        pc (torch.Tensor): Tensor of shape [BS, N, 6] (xyz + normals).

    Returns:
        torch.Tensor: Rotated point clouds of shape [BS, N, 6].
    """
    BS, N, _ = pc.shape
    device = pc.device
    # Generate random Euler angles [BS, 3]
    angles = torch.rand(BS, 3, device=device) * 2 * torch.pi  # [BS, (rx, ry, rz)]
    cos = torch.cos
    sin = torch.sin
    rx, ry, rz = angles[:, 0], angles[:, 1], angles[:, 2]
    # Compute rotation matrices for each axis
    ones = torch.ones(BS, device=device)
    zeros = torch.zeros(BS, device=device)
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, cos(rx), -sin(rx)], dim=1),
        torch.stack([zeros, sin(rx),  cos(rx)], dim=1)
    ], dim=1)  # [BS, 3, 3]
    Ry = torch.stack([
        torch.stack([ cos(ry), zeros, sin(ry)], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-sin(ry), zeros, cos(ry)], dim=1)
    ], dim=1)
    Rz = torch.stack([
        torch.stack([cos(rz), -sin(rz), zeros], dim=1),
        torch.stack([sin(rz),  cos(rz), zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)
    # Combine rotations: R = Rz @ Ry @ Rx
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)  # [BS, 3, 3]
    # Split positions and normals
    xyz = pc[..., :3]  # [BS, N, 3]
    normals = pc[..., 3:]  # [BS, N, 3]
    # Apply rotation: reshape to [BS, 3, N], then matmul
    xyz_rotated = torch.bmm(xyz, R.transpose(1, 2))  # [BS, N, 3]
    normals_rotated = torch.bmm(normals, R.transpose(1, 2))  # [BS, N, 3]
    # Concatenate back to [BS, N, 6]
    rotated_pc = torch.cat([xyz_rotated, normals_rotated], dim=2)
    return rotated_pc

def center(pc: torch.Tensor) -> torch.Tensor:
    """
    Centers each point cloud in the batch to the origin by subtracting the mean position.
    
    Args:
        pc (torch.Tensor): Tensor of shape [BS, N, 6] (xyz + normals).

    Returns:
        torch.Tensor: Centered point clouds of shape [BS, N, 6].
    """
    xyz = pc[:, :, :3]  # [BS, N, 3]
    normals = pc[:, :, 3:]  # [BS, N, 3]
    # Compute the centroid of each point cloud
    centroid = xyz.mean(dim=1, keepdim=True)  # [BS, 1, 3]
    # Subtract the centroid to center the cloud
    centered_xyz = xyz - centroid  # [BS, N, 3]
    # Recombine positions and normals
    centered_pc = torch.cat([centered_xyz, normals], dim=2)  # [BS, N, 6]
    return centered_pc   

def scale(pc: torch.Tensor) -> torch.Tensor:
    """
    Scales each point cloud in the batch uniformly so it fits inside the bounding box [-1, 1]^3.

    Args:
        pc (torch.Tensor): Tensor of shape [BS, N, 6] (xyz + normals).

    Returns:
        torch.Tensor: Scaled point clouds of shape [BS, N, 6].
    """
    bbox_min = torch.tensor([-1.0, -1.0, -1.0], device=pc.device)
    bbox_max = torch.tensor([1.0, 1.0, 1.0], device=pc.device)
    xyz = pc[:, :, :3]  # [BS, N, 3]
    normals = pc[:, :, 3:]  # [BS, N, 3]
    # Find current min/max per batch
    current_min = xyz.min(dim=1, keepdim=True).values  # [BS, 1, 3]
    current_max = xyz.max(dim=1, keepdim=True).values  # [BS, 1, 3]
    # Compute the size of current bounding box
    extent = current_max - current_min  # [BS, 1, 3]
    max_extent = extent.max(dim=2, keepdim=True).values  # [BS, 1, 1]
    # Compute the target size of the bbox
    target_extent = (bbox_max - bbox_min).max()  # scalar, = 2.0
    # Compute the uniform scale factor per point cloud
    scale_factor = target_extent / max_extent  # [BS, 1, 1]
    # Apply scale to positions
    scaled_xyz = xyz * scale_factor  # [BS, N, 3]
    # Combine back with normals
    scaled_pc = torch.cat([scaled_xyz, normals], dim=2)  # [BS, N, 6]
    return scaled_pc

def parts_augmentations(pc, rotation_enabled=False, scale_parts=False):
    """
    Augments each point cloud in the batch using centering, optional rotation, and scaling.

    Args:
        pc (torch.Tensor): Tensor of shape [BS, N, 6] (xyz + normals).
        rotation_enabled (bool): Whether to apply random rotation on each point cloud.

    Returns:
        torch.Tensor: Augmented point clouds.
                      If rotation_enabled is True, shape is [BS * 2, N, 6].
                      If False, shape is [BS, N, 6].
    """
    # Always center first
    pc = center(pc)

    if rotation_enabled:
        # Apply rotation only to a copy of the centered pc
        rotated = rotation(pc)
        pc = torch.cat([pc, rotated], dim=0)

        # Apply scaling to both original and rotated
    if scale_parts:
        scaled = scale(pc)
        pc = torch.cat([pc, scaled], dim=0)
    
    return pc
