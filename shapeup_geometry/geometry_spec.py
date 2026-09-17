"""Architecture and sampling spec for the geometry pipeline.

Kept in the package so the inference script and the demo share one definition,
and so neither depends on the training yamls under configs/ (those are for
training and must be free to change).
"""

# Every architecture and sampling value the pipeline needs, pinned here: the
# configs/ yamls are for training and this script must not drift when they change.
GEOMETRY_SPEC = {
    # use_lora gates the adapter: without it you get base Step1X-3D, not ShapeUp.
    "use_lora": True,
    "n_part_latents": 1024,
    "cfg_type": "separate_visual",
    "guidance_scale": 3.5,
    "bounds": 1.05,
    "mc_level": 0.0,
    "shape_model_type": "michelangelo-autoencoder",
    "shape_model": {
        "pretrained_model_name_or_path": "stepfun-ai/Step1X-3D",
        "subfolder": "Step1X-3D-Geometry-Label-1300m",
        "n_samples": 32768,        # was ${data.n_samples}
        "with_sharp_data": True,   # was ${data.with_sharp_data}
        "use_downsample": True,
        "num_latents": 2048,
        "embed_dim": 64,
        "point_feats": 3,
        "out_dim": 1,
        "num_freqs": 8,
        "include_pi": False,
        "heads": 12,
        "width": 768,
        "num_encoder_layers": 8,
        "num_decoder_layers": 16,
        "use_ln_post": True,
        "init_scale": 0.25,
        "qkv_bias": False,
        "use_flash": True,
        "use_checkpoint": False,
    },
    "visual_condition_type": "dinov2-clip-encoder",
    "visual_condition": {
        "pretrained_dino_name_or_path": "facebook/dinov2-with-registers-large",
        "pretrained_clip_name_or_path": "openai/clip-vit-large-patch14",
        "encode_camera": False,
        "n_views": 1,              # was ${data.n_views}
        "empty_embeds_ratio": 0.1,
        "normalize_embeds": False,
        "zero_uncond_embeds": True,
        "image_size": 224,
    },
    "denoiser_model_type": "flux-denoiser",
    "denoiser_model": {
        "pretrained_model_name_or_path": "stepfun-ai/Step1X-3D",
        "subfolder": "Step1X-3D-Geometry-Label-1300m",
        "input_channels": 64,      # was ${system.shape_model.embed_dim}
        "width": 1536,
        "layers": 8,
        "num_single_layers": 16,
        "num_heads": 16,
        "use_visual_condition": True,
        "visual_condition_dim": 1024,
        "n_views": 1,              # was ${data.n_views}
        "use_shape_condition": True,
        "shape_condition_dim": 64,  # was ${system.shape_model.embed_dim}
    },
    "denoise_scheduler_type": "diffusers.schedulers.FlowMatchEulerDiscreteScheduler",
    "denoise_scheduler": {"num_train_timesteps": 1000},
}
