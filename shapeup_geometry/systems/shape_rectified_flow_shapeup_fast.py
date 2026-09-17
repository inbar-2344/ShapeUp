from dataclasses import dataclass, field
import open3d as o3d
import numpy as np
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from einops import repeat
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import matplotlib.image as mpimg
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
import shapeup_geometry
from shapeup_geometry.systems.base import BaseSystem
from shapeup_geometry.utils.misc import get_rank
from shapeup_geometry.utils.typing import *
from shapeup_geometry.systems.utils import read_image, preprocess_image, flow_sample, flow_sample_separate_cfg
from shapeup_geometry.models.pipelines.pipeline_utils import smart_load_model
from safetensors.torch import load_file
import wandb

def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

@shapeup_geometry.register("shapeup-fast-rectified-flow-system")
class ShapeupRectifiedFlowSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        skip_validation: bool = False
        bounds: float = 1.05
        mc_level: float = 0.0
        octree_resolution: int = 256

        # diffusion config
        cfg_type: str = "single" # single, separate
        guidance_scale: float = 7.5
        guidance_scale_visual: float = 1.5
        guidance_scale_shape: float = 2.5
        num_inference_steps: int = 30
        eta: float = 0.0
        snr_gamma: float = 5.0

        # flow
        weighting_scheme: str = "logit_normal"
        logit_mean: float = 0
        logit_std: float = 1.0
        mode_scale: float = 1.29
        precondition_outputs: bool = True
        precondition_t: int = 1000

        # shape vae model
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)

        # condition model
        visual_condition_type: Optional[str] = None
        visual_condition: dict = field(default_factory=dict)
        caption_condition_type: Optional[str] = None
        caption_condition: dict = field(default_factory=dict)
        label_condition_type: Optional[str] = None
        label_condition: dict = field(default_factory=dict)

        # diffusion model
        denoiser_model_type: str = None
        denoiser_model: dict = field(default_factory=dict)

        # noise scheduler
        noise_scheduler_type: str = None
        noise_scheduler: dict = field(default_factory=dict)

        # denoise scheduler
        denoise_scheduler_type: str = None
        denoise_scheduler: dict = field(default_factory=dict)

        # lora
        use_lora: bool = False
        lora_layers: Optional[str] = None
        rank: int = 128  # The dimension of the LoRA update matrices.
        alpha: int = 128
        
        # part cond
        n_part_latents: int = 256
        skip_visual_cond: bool = False
        
    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = shapeup_geometry.find(self.cfg.shape_model_type)(
            self.cfg.shape_model
        )
        self.shape_model.eval()
        self.shape_model.requires_grad_(False)

        if self.cfg.visual_condition_type is not None:
            self.visual_condition = shapeup_geometry.find(
                self.cfg.visual_condition_type
            )(self.cfg.visual_condition)
            self.visual_condition.requires_grad_(False)

        if self.cfg.caption_condition_type is not None:
            self.caption_condition = shapeup_geometry.find(
                self.cfg.caption_condition_type
            )(self.cfg.caption_condition)
            self.caption_condition.requires_grad_(False)

        if self.cfg.label_condition_type is not None:
            self.label_condition = shapeup_geometry.find(
                self.cfg.label_condition_type
            )(self.cfg.label_condition)
            
        self.denoiser_model = shapeup_geometry.find(self.cfg.denoiser_model_type)(
            self.cfg.denoiser_model
        )

        self.noise_scheduler = shapeup_geometry.find(self.cfg.noise_scheduler_type)(
            **self.cfg.noise_scheduler
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        self.denoise_scheduler = shapeup_geometry.find(
            self.cfg.denoise_scheduler_type
        )(**self.cfg.denoise_scheduler)

        if self.cfg.use_lora:
            from peft import LoraConfig, set_peft_model_state_dict
            self.denoiser_model.dit_model.requires_grad_(False)
            if self.cfg.lora_layers is not None:
                self.target_modules = [
                    layer.strip() for layer in self.cfg.lora_layers.split(",")
                ]
            else:
                self.target_modules = [
                    "attn.to_k",
                    "attn.to_q",
                    "attn.to_v",
                    "attn.to_out.0",
                    "attn.add_k_proj",
                    "attn.add_q_proj",
                    "attn.add_v_proj",
                    "attn.to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                ]
                self.transformer_lora_config = LoraConfig(
                    r=self.cfg.rank,
                    lora_alpha=self.cfg.alpha,
                    init_lora_weights="gaussian",
                    target_modules=self.target_modules,
                )
                self.denoiser_model.dit_model.add_adapter(self.transformer_lora_config)

    def sample_posterior(self, posterior_mean, posterior_std):
        kl_embed = posterior_mean + posterior_std * torch.randn_like(posterior_mean)
        # kl_embed = kl_embed * self.cfg.z_scale_factor # z_scale_factor is set to 1.0
        return kl_embed

    def forward(self, batch: Dict[str, Any], skip_noise=False) -> Dict[str, Any]:
        # 1. sample shape latents from shape posterior
        latents = self.sample_posterior(batch["shape_posterior_mean"], batch["shape_posterior_std"])
        bs, n_latents, latent_dim = batch["obj_posterior_mean"].shape
        part_latents = self.sample_posterior(batch["obj_posterior_mean"], batch["obj_posterior_std"])
        # choose n_part_latents random latents for each part in batch 
        rand_scores = torch.rand(bs, n_latents, device=part_latents.device)
        _, indices = torch.sort(rand_scores, dim=1)
        indices = indices[:, :self.cfg.n_part_latents]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, latent_dim)
        shape_cond = torch.gather(part_latents, dim=1, index=indices_expanded)
        shape_cond = shape_cond.view(bs, self.cfg.n_part_latents, latent_dim) # (bs, 512, 64)
        if self.training:
            mask = (torch.rand(bs) > 0.1).float().view(bs, 1, 1).to(shape_cond)
            shape_cond = shape_cond * mask
            # if random.random() < 0.1: # support cfg mode
            #     shape_cond = torch.zeros_like(shape_cond)
    
        # 2. gain visual condition # Texture/Constant chair image
        visual_cond = None
        n_images = 1
        if not self.cfg.skip_visual_cond:
            visual_cond = self.visual_condition(batch).to(latents)
            bs = visual_cond.shape[0]
            

        # 3. sample noise that we"ll add to the latents
        noise = torch.randn_like(latents).to(
            latents
        )  # [batch_size, n_token, latent_dim]

        # 4. Sample a random timestep
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.cfg.weighting_scheme,
            batch_size=bs * n_images,
            logit_mean=self.cfg.logit_mean,
            logit_std=self.cfg.logit_std,
            mode_scale=self.cfg.mode_scale,
        )
        indices = (u * self.cfg.noise_scheduler.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(
            device=latents.device
        )

        # 5. add noise
        sigmas = get_sigmas(
            self.noise_scheduler_copy, timesteps, n_dim=3, dtype=latents.dtype
        )
        noisy_z = (1.0 - sigmas) * latents + sigmas * noise

        # 6. diffusion model forward
        output = self.denoiser_model(
            model_input=noisy_z, timestep=timesteps.long(), visual_condition=visual_cond, 
            caption_condition=None, label_condition=None, shape_condition=shape_cond).sample

        # 7. compute loss
        if self.cfg.precondition_outputs:
            output = output * (-sigmas) + noisy_z
        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.cfg.weighting_scheme, sigmas=sigmas
        )
        # flow matching loss
        if self.cfg.precondition_outputs:
            target = latents
        else:
            target = noise - latents

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (output.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()

        return {
            "loss_diffusion": loss,
            "latents": latents,
            "x_t": noisy_z,
            "noise": noise,
            "noise_pred": output,
            "timesteps": timesteps,
        }

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.0
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
            if name.startswith("log_"):
                self.log(f"log/{name.replace('log_', '')}", value.mean())

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_"):
                self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}
    
    @torch.no_grad()
    def plot_and_save_shapes_ply(self, input_image, gt_image, points_wo_cfg, points_w_cfg, uid, input_shape_surface, gt_shape_surface, input_shape_name, gt_shape_name, wandb_logger=None):
        n_plots = 6
        fig = plt.figure(figsize=(6 * n_plots, 6))
        axes = []
        # input image subplot
        input_img = fig.add_subplot(1, n_plots, 1)  # index 1-based
        # img = mpimg.imread(image)
        input_img.imshow(input_image)
        input_img.axis('off')
        input_img.set_title("Cond Image")
        axes.append(input_img)

        gt_img = fig.add_subplot(1, n_plots, 2)  # index 1-based
        # img = mpimg.imread(image)
        gt_img.imshow(gt_image)
        gt_img.axis('off')
        gt_img.set_title("GT Image")
        axes.append(gt_img)


        # Add the 3D subplots starting from index=2
        for i in range(3, 7):
            ax = fig.add_subplot(1, n_plots, i, projection='3d')
            axes.append(ax)
        # Plot wo_cfg

        axes[2].scatter(points_wo_cfg[:, 0], points_wo_cfg[:, 1], points_wo_cfg[:, 2], zdir='z', s=0.5, c=points_wo_cfg[:, 2], cmap='jet')
        axes[2].view_init(elev=-90, azim=90)
        axes[2].set_title(f"without cfg")
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")
        axes[2].set_zlabel("Z")
        # Plot w_cfg
        axes[3].scatter(points_w_cfg[:, 0], points_w_cfg[:, 1], points_w_cfg[:, 2], zdir='z' ,s=0.5, c=points_w_cfg[:, 2], cmap='jet')
        axes[3].view_init(elev=-90, azim=90)
        axes[3].set_title(f"with cfg")
        axes[3].set_xlabel("X")
        axes[3].set_ylabel("Y")
        axes[3].set_zlabel("Z")

        # Plot input shape pc 
        axes[4].scatter(input_shape_surface[:, 0], input_shape_surface[:, 1], input_shape_surface[:, 2], zdir='z' ,s=0.5, c=input_shape_surface[:, 2], cmap='jet')
        axes[4].view_init(elev=-90, azim=90)
        axes[4].set_title(f"Input shape: {input_shape_name}")
        axes[4].set_xlabel("X")
        axes[4].set_ylabel("Y")
        axes[4].set_zlabel("Z")

        # Plot GT shape pc 
        axes[5].scatter(gt_shape_surface[:, 0], gt_shape_surface[:, 1], gt_shape_surface[:, 2], zdir='z' ,s=0.5, c=gt_shape_surface[:, 2], cmap='jet')
        axes[5].view_init(elev=-90, azim=90)
        axes[5].set_title(f"GT shape: {gt_shape_name}")
        axes[5].set_xlabel("X")
        axes[5].set_ylabel("Y")
        axes[5].set_zlabel("Z")


        # Optional: global title
        fig.suptitle(f"{uid} - Step {self.true_global_step}", fontsize=16)
        # Save combined figure and part pc
        fig_path = self.get_save_path(f"it{self.true_global_step}/{uid}/ps.png")
        plt.savefig(fig_path)

        input_shape_path = self.get_save_path(f"it{self.true_global_step}/{uid}/{input_shape_name}.ply")
        xyz = input_shape_surface[..., :3]
        normals = input_shape_surface[..., 3:]
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
        pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals, dtype=np.float64))
        o3d.io.write_point_cloud(input_shape_path, pcd)

        gt_shape_path = self.get_save_path(f"it{self.true_global_step}/{uid}/{input_shape_name}.ply")
        xyz = gt_shape_surface[..., :3]
        normals = gt_shape_surface[..., 3:]
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
        pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals, dtype=np.float64))
        o3d.io.write_point_cloud(gt_shape_path, pcd)

        # Log to wandb
        if wandb_logger is not None:
            wandb_logger.experiment.log({f"{uid}": wandb.Image(fig), "step": self.true_global_step})
        plt.close(fig)
    
    
    @torch.no_grad()
    def plot_and_save_shapes_ply_test(self, input_image, points_wo_cfg, points_w_cfg, uid, input_shape_surface, input_shape_name, wandb_logger=None):
        n_plots = 4
        fig = plt.figure(figsize=(6 * n_plots, 6))
        axes = []
        # input image subplot
        input_img = fig.add_subplot(1, n_plots, 1)  # index 1-based
        # img = mpimg.imread(image)
        input_img.imshow(input_image)
        input_img.axis('off')
        input_img.set_title("Cond Image")
        axes.append(input_img)


        # Add the 3D subplots starting from index=2
        for i in range(2, 5):
            ax = fig.add_subplot(1, n_plots, i, projection='3d')
            axes.append(ax)
        # Plot wo_cfg

        axes[1].scatter(points_wo_cfg[:, 0], points_wo_cfg[:, 1], points_wo_cfg[:, 2], zdir='z', s=0.5, c=points_wo_cfg[:, 2], cmap='jet')
        axes[1].view_init(elev=-90, azim=90)
        axes[1].set_title(f"without cfg")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        axes[1].set_zlabel("Z")
        # Plot w_cfg
        axes[2].scatter(points_w_cfg[:, 0], points_w_cfg[:, 1], points_w_cfg[:, 2], zdir='z' ,s=0.5, c=points_w_cfg[:, 2], cmap='jet')
        axes[2].view_init(elev=-90, azim=90)
        axes[2].set_title(f"with cfg")
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")
        axes[2].set_zlabel("Z")

        # Plot input shape pc 
        axes[3].scatter(input_shape_surface[:, 0], input_shape_surface[:, 1], input_shape_surface[:, 2], zdir='z' ,s=0.5, c=input_shape_surface[:, 2], cmap='jet')
        axes[3].view_init(elev=-90, azim=90)
        axes[3].set_title(f"Input shape: {input_shape_name}")
        axes[3].set_xlabel("X")
        axes[3].set_ylabel("Y")
        axes[3].set_zlabel("Z")

        # Optional: global title
        fig.suptitle(f"{uid} - Step {self.true_global_step}", fontsize=16)
        # Save combined figure and part pc
        fig_path = self.get_save_path(f"it{self.true_global_step}/{uid}/ps.png")
        plt.savefig(fig_path)

        input_shape_path = self.get_save_path(f"it{self.true_global_step}/{uid}/{input_shape_name}.ply")
        xyz = input_shape_surface[..., :3]
        normals = input_shape_surface[..., 3:]
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
        pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals, dtype=np.float64))
        o3d.io.write_point_cloud(input_shape_path, pcd)

        # Log to wandb
        if wandb_logger is not None:
            wandb_logger.experiment.log({f"{uid}": wandb.Image(fig), "step": self.true_global_step})
        plt.close(fig)
    
    @torch.no_grad()
    def validation_step_old(self, batch, batch_idx):
        wandb_logger = self.loggers[2]
        if self.cfg.skip_validation:
            print("skipping validation")
            return {}
        bs, n_latents, latent_dim = batch["obj_posterior_mean"].shape
        obj_latents = self.sample_posterior(batch["obj_posterior_mean"], batch["obj_posterior_std"])
        rand_scores = torch.rand(bs , n_latents, device=obj_latents.device)
        _, indices = torch.sort(rand_scores, dim=1)
        indices = indices[:, :self.cfg.n_part_latents]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, latent_dim)
        shape_cond = torch.gather(obj_latents, dim=1, index=indices_expanded)
        shape_cond = shape_cond.view(bs, self.cfg.n_part_latents, latent_dim) # (bs, 256, 64)
        if not self.cfg.skip_visual_cond:
            sample_inputs = {
                "image": batch["input_image"],
                "latents": shape_cond
            }
        else:
            sample_inputs = {
                "latents": shape_cond
            }
            
        sample_outputs_cfg = self.sample(sample_inputs)  # list
        sample_outputs_wo_cfg = self.sample(sample_inputs, guidance_scale=1.0)  # list
        repetitions = len(sample_outputs_cfg["latents"])
        for rep in range(repetitions):
            meshes_cfg = self.shape_model.extract_geometry(
                sample_outputs_cfg["latents"][rep],
                bounds=self.cfg.bounds,
                mc_level=self.cfg.mc_level,
                octree_resolution=self.cfg.octree_resolution,
                enable_pbar=False,
            )
            meshes_wo_cfg = self.shape_model.extract_geometry(
                sample_outputs_wo_cfg["latents"][rep],
                bounds=self.cfg.bounds,
                mc_level=self.cfg.mc_level,
                octree_resolution=self.cfg.octree_resolution,
                enable_pbar=False,
            )
            len_meshes_cfg = len(meshes_cfg)
            for j in range(len_meshes_cfg):
                name = f'{batch["uid"][j]}_rep{rep}'
                input_shape_surface = batch["input_shape_surface"][j].to(dtype=torch.float32).detach().cpu().numpy()[0]
                gt_shape_surface = batch["gt_shape_surface"][j].to(dtype=torch.float32).detach().cpu().numpy()[0]
                input_shape_index = batch["input_shape_index"][j]
                input_shape_name = f"Input shape index: {input_shape_index.item()}"
                gt_shape_index = batch["gt_shape_index"][j]
                gt_shape_name = f"GT shape index: {gt_shape_index.item()}"
                if (
                    meshes_cfg[j].verts is not None
                    and meshes_cfg[j].verts.shape[0] > 0
                    and meshes_cfg[j].faces is not None
                    and meshes_cfg[j].faces.shape[0] > 0
                ):
                    self.save_mesh(
                        f"it{self.true_global_step}/{name}/w_cfg.obj",
                        meshes_cfg[j].verts,
                        meshes_cfg[j].faces,
                    )
                    self.save_mesh(
                        f"it{self.true_global_step}/{name}/wo_cfg.obj",
                        meshes_wo_cfg[j].verts,
                        meshes_wo_cfg[j].faces,
                    )
                    # Plot and save part point cloud
                    # Extract points for both
                    points_wo = meshes_wo_cfg[j].verts.detach().cpu().numpy()
                    points_w = meshes_cfg[j].verts.detach().cpu().numpy()
                    if not self.cfg.skip_visual_cond:
                        input_image = batch["input_image"][j].to(dtype=torch.float32).detach().cpu().numpy()
                        original_shape_image = batch["original_shape_image"][j].to(dtype=torch.float32).detach().cpu().numpy()
                    else:
                        input_image = None
                    self.plot_and_save_shapes_ply(input_image, original_shape_image, points_wo, points_w, name, input_shape_surface, gt_shape_surface, input_shape_name, gt_shape_name, wandb_logger)
                    
        out = self(batch)
        self.log(f"val/loss", out["loss_diffusion"])
        return {"val/loss": out["loss_diffusion"]}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        wandb_logger = self.loggers[2]
        if self.cfg.skip_validation:
            print("skipping validation")
            return {}
        n_parts, N = batch["objs_surface"][:, 0].shape[:2]
        objs_surface = batch["objs_surface"][:, 0]
        objs_sharp_surface = batch["objs_sharp_surface"][:, 0]
        input_shape, kl_embed, posterior = self.shape_model.encode(
        objs_surface,
        sharp_surface=objs_sharp_surface,
        sample_posterior=True,
        )

        bs, n_latents, latent_dim = kl_embed.shape
        rand_scores = torch.rand(bs , n_latents, device=kl_embed.device)
        _, indices = torch.sort(rand_scores, dim=1)
        indices = indices[:, :self.cfg.n_part_latents]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, latent_dim)
        shape_cond = torch.gather(kl_embed, dim=1, index=indices_expanded)
        shape_cond = shape_cond.view(bs, self.cfg.n_part_latents, latent_dim) # (bs, n_part_latents, 64)
        
        if not self.cfg.skip_visual_cond:
            sample_inputs = {
                "image": batch["input_image"],
                "latents": shape_cond
            }
        else:
            sample_inputs = {
                "latents": shape_cond
            }
            
        # Same guidance scheme as test_step / inference, so validation meshes are
        # comparable to what the inference pipeline produces.
        if self.cfg.cfg_type == "single":
            sample_outputs_cfg = self.sample(sample_inputs, guidance_scale=self.cfg.guidance_scale)  # list
        else:
            sample_outputs_cfg = self.sample_separate_cfg(sample_inputs, guidance_scale_visual=self.cfg.guidance_scale_visual, guidance_scale_shape=self.cfg.guidance_scale_shape, cfg_type=self.cfg.cfg_type)  # list
        sample_outputs_wo_cfg = self.sample(sample_inputs, guidance_scale=1.0)  # list
        repetitions = len(sample_outputs_cfg["latents"])
        for rep in range(repetitions):
            meshes_cfg = self.shape_model.extract_geometry(
                sample_outputs_cfg["latents"][rep],
                bounds=self.cfg.bounds,
                mc_level=self.cfg.mc_level,
                octree_resolution=self.cfg.octree_resolution,
                enable_pbar=False,
            )
            meshes_wo_cfg = self.shape_model.extract_geometry(
                sample_outputs_wo_cfg["latents"][rep],
                bounds=self.cfg.bounds,
                mc_level=self.cfg.mc_level,
                octree_resolution=self.cfg.octree_resolution,
                enable_pbar=False,
            )
            len_meshes_cfg = len(meshes_cfg)
            for j in range(len_meshes_cfg):
                name = f'{batch["uid"][j]}_rep{rep}'
                input_shape_surface = batch["objs_surface"][j].to(dtype=torch.float32).detach().cpu().numpy()[0]
                input_shape_name = "Input shape pc"

                if (
                    meshes_cfg[j].verts is not None
                    and meshes_cfg[j].verts.shape[0] > 0
                    and meshes_cfg[j].faces is not None
                    and meshes_cfg[j].faces.shape[0] > 0
                ):
                    self.save_mesh(
                        f"it{self.true_global_step}/{name}/w_cfg.obj",
                        meshes_cfg[j].verts,
                        meshes_cfg[j].faces,
                    )
                    self.save_mesh(
                        f"it{self.true_global_step}/{name}/wo_cfg.obj",
                        meshes_wo_cfg[j].verts,
                        meshes_wo_cfg[j].faces,
                    )
                    # Plot and save part point cloud
                    # Extract points for both
                    points_wo = meshes_wo_cfg[j].verts.detach().cpu().numpy()
                    points_w = meshes_cfg[j].verts.detach().cpu().numpy()
                    if not self.cfg.skip_visual_cond:
                        input_image = batch["input_image"][j].to(dtype=torch.float32).detach().cpu().numpy()
                    else:
                        input_image = None
                    self.plot_and_save_shapes_ply_test(input_image, points_wo, points_w, name, input_shape_surface, input_shape_name, wandb_logger=wandb_logger)
                    
        # out = self(batch)
        # self.log(f"val/loss", out["loss_diffusion"])
        # return {"val/loss": out["loss_diffusion"]}

    @torch.no_grad()
    def sample(
        self,
        sample_inputs: Dict[str, Union[torch.FloatTensor, List[str]]],
        sample_times: int = 1,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        eta: float = 0.0,
        seed: Optional[int] = None,
        **kwargs,
    ):

        if steps is None:
            steps = self.cfg.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.cfg.guidance_scale
        do_classifier_free_guidance = guidance_scale != 1.0

        # conditional encode
        visal_cond = None
        if "image" in sample_inputs:
            if type(sample_inputs["image"][0]) == str:
                sample_inputs["image"] = [Image.open(img) for img in sample_inputs["image"]]
                sample_inputs["image"] = preprocess_image(sample_inputs["image"], **kwargs)
                
            visal_cond = self.visual_condition.encode_image(sample_inputs["image"], **kwargs)
            if do_classifier_free_guidance:
                un_cond = self.visual_condition.empty_image_embeds.repeat(
                    len(sample_inputs["image"]), 1, 1).to(visal_cond)
                #un_cond = visal_cond.clone()
                visal_cond = torch.cat([un_cond, visal_cond], dim=0)
        caption_cond = None
        label_cond = None
        shape_cond = None
        if "latents" in sample_inputs:
            shape_cond = sample_inputs["latents"]
            if do_classifier_free_guidance:
                un_cond = torch.zeros_like(sample_inputs["latents"]).to(shape_cond)
                #un_cond = shape_cond.clone()
                shape_cond = torch.cat([un_cond, shape_cond], dim=0)

        latents_list = []
        print(f"sampling with seed: {seed}")
        if seed != None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None

        for _ in range(sample_times):
            sample_loop = flow_sample(
                self.denoise_scheduler,
                self.denoiser_model.eval(),
                shape=self.shape_model.latent_shape,
                visual_cond=visal_cond,
                caption_cond=caption_cond,
                label_cond=label_cond,
                shape_cond=shape_cond,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.device,
                eta=eta,
                disable_prog=False,
                generator=generator,
            )
            for sample, t in sample_loop:
                latents = sample
            latents_list.append(self.shape_model.decode(latents))

        return {"latents": latents_list, "inputs": sample_inputs}

    def on_validation_epoch_end(self):
        pass
    

    @torch.no_grad()
    def sample_separate_cfg(
        self,
        sample_inputs: Dict[str, Union[torch.FloatTensor, List[str]]],
        sample_times: int = 1,
        steps: Optional[int] = None,
        guidance_scale_visual: Optional[float] = 7.5,
        guidance_scale_shape: Optional[float] = 2.5,
        eta: float = 0.0,
        seed: Optional[int] = None,
        cfg_type: str = "separate_visual",
        **kwargs,
    ):

        if steps is None:
            steps = self.cfg.num_inference_steps
        if guidance_scale_visual is None:
            guidance_scale = self.cfg.guidance_scale
        do_classifier_free_guidance = guidance_scale_visual != 1.0 and guidance_scale_shape != 1.0

        # conditional encode
        visal_cond = None
        if "image" in sample_inputs:
            if type(sample_inputs["image"][0]) == str:
                sample_inputs["image"] = [Image.open(img) for img in sample_inputs["image"]]
                sample_inputs["image"] = preprocess_image(sample_inputs["image"], **kwargs)
                
            visal_cond = self.visual_condition.encode_image(sample_inputs["image"], **kwargs)
            if do_classifier_free_guidance:
                un_cond = self.visual_condition.empty_image_embeds.repeat(
                    len(sample_inputs["image"]), 1, 1).to(visal_cond)
                #un_cond = visal_cond.clone()
                if cfg_type == "separate":
                    visal_cond = torch.cat([un_cond, un_cond, visal_cond], dim=0)
                else: # separate_visual
                    visal_cond = torch.cat([un_cond, visal_cond, visal_cond], dim=0)
        caption_cond = None
        label_cond = None
        shape_cond = None
        if "latents" in sample_inputs:
            shape_cond = sample_inputs["latents"]
            if do_classifier_free_guidance:
                un_cond = torch.zeros_like(sample_inputs["latents"]).to(shape_cond)
                #un_cond = shape_cond.clone()
                if cfg_type == "separate":
                    shape_cond = torch.cat([un_cond, shape_cond, shape_cond], dim=0)
                else: # separate_visual
                    shape_cond = torch.cat([un_cond, un_cond, shape_cond], dim=0)

        latents_list = []
        if seed != None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None

        for _ in range(sample_times):
            sample_loop = flow_sample_separate_cfg(
                self.denoise_scheduler,
                self.denoiser_model.eval(),
                shape=self.shape_model.latent_shape,
                visual_cond=visal_cond,
                caption_cond=caption_cond,
                label_cond=label_cond,
                shape_cond=shape_cond,
                steps=steps,
                guidance_scale_visual=guidance_scale_visual,
                guidance_scale_shape=guidance_scale_shape,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.device,
                eta=eta,
                disable_prog=False,
                generator=generator,
            )
            for sample, t in sample_loop:
                latents = sample
            latents_list.append(self.shape_model.decode(latents))

        return {"latents": latents_list, "inputs": sample_inputs}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        torch.manual_seed(10)  # Reset to same state every batch
        torch.cuda.manual_seed(10)
        n_parts, N = batch["objs_surface"][:, 0].shape[:2]
        objs_surface = batch["objs_surface"][:, 0]
        objs_sharp_surface = batch["objs_sharp_surface"][:, 0]
        input_shape, kl_embed, posterior = self.shape_model.encode(
        objs_surface,
        sharp_surface=objs_sharp_surface,
        sample_posterior=True,
        )
        input_shape_latents = self.shape_model.decode(kl_embed)
        bs, n_latents, latent_dim = kl_embed.shape
        rand_scores = torch.rand(bs , n_latents, device=kl_embed.device)
        _, indices = torch.sort(rand_scores, dim=1)
        indices = indices[:, :self.cfg.n_part_latents]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, latent_dim)
        shape_cond = torch.gather(kl_embed, dim=1, index=indices_expanded)
        shape_cond = shape_cond.view(bs, self.cfg.n_part_latents, latent_dim) # (bs, 256, 64)
        if not self.cfg.skip_visual_cond:
            sample_inputs = {
                "image": batch["input_image"],
                "latents": shape_cond
            }
        else:
            sample_inputs = {
                "latents": shape_cond
            }
        if self.cfg.cfg_type == "single":
            sample_outputs_cfg = self.sample(sample_inputs, guidance_scale=self.cfg.guidance_scale)  # list
        else:
            sample_outputs_cfg = self.sample_separate_cfg(sample_inputs, guidance_scale_visual=self.cfg.guidance_scale_visual, guidance_scale_shape=self.cfg.guidance_scale_shape, cfg_type=self.cfg.cfg_type)  # list
        sample_outputs_wo_cfg = self.sample(sample_inputs, guidance_scale=1.0)  # list
        repetitions = len(sample_outputs_cfg["latents"])

        for rep in range(repetitions):
            input_shape_meshes = self.shape_model.extract_geometry(
                input_shape_latents,
                bounds=self.cfg.bounds,
                mc_level=self.cfg.mc_level,
                octree_resolution=self.cfg.octree_resolution,
                enable_pbar=False,
            )

            meshes_cfg = self.shape_model.extract_geometry(
                sample_outputs_cfg["latents"][rep],
                bounds=self.cfg.bounds,
                mc_level=self.cfg.mc_level,
                octree_resolution=self.cfg.octree_resolution,
                enable_pbar=False,
            )
            meshes_wo_cfg = self.shape_model.extract_geometry(
                sample_outputs_wo_cfg["latents"][rep],
                bounds=self.cfg.bounds,
                mc_level=self.cfg.mc_level,
                octree_resolution=self.cfg.octree_resolution,
                enable_pbar=False,
            )
            
            len_meshes_cfg = len(meshes_cfg)
            for j in range(len_meshes_cfg):
                name = f'{batch["uid"][j]}_img_{batch["input_image_name"][j]}'
                input_shape_surface = batch["objs_surface"][j].to(dtype=torch.float32).detach().cpu().numpy()[0]
                input_shape_name = "Input shape pc"

                
                if (
                    meshes_cfg[j].verts is not None
                    and meshes_cfg[j].verts.shape[0] > 0
                    and meshes_cfg[j].faces is not None
                    and meshes_cfg[j].faces.shape[0] > 0
                ):

                    self.save_mesh(
                        f"it{self.true_global_step}/{name}/input_shape.obj",
                        input_shape_meshes[j].verts,
                        input_shape_meshes[j].faces,
                    )
                    self.save_mesh(
                        f"it{self.true_global_step}/{name}/wo_cfg.obj",
                        meshes_wo_cfg[j].verts,
                        meshes_wo_cfg[j].faces,
                    )
                    self.save_mesh(
                        f"it{self.true_global_step}/{name}/w_cfg.obj",
                        meshes_cfg[j].verts,
                        meshes_cfg[j].faces,
                    )
                    
                    points_wo = meshes_wo_cfg[j].verts.detach().cpu().numpy()
                    points_w = meshes_cfg[j].verts.detach().cpu().numpy()
                    if not self.cfg.skip_visual_cond:
                        input_image = batch["input_image"][j].to(dtype=torch.float32).detach().cpu().numpy()
                    else:
                        image=None
                    self.plot_and_save_shapes_ply_test(input_image, points_wo, points_w, name, input_shape_surface, input_shape_name)
                        


