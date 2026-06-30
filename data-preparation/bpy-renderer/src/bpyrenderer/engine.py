from typing import Literal

import bpy
import logging
logging.getLogger("blender").setLevel(logging.ERROR)

def init_render_engine(
    engine: Literal["CYCLES", "BLENDER_EEVEE"], render_samples: int = 16
):
    """Initialize the rendering engine.

    Args:
        engine (Literal[&quot;CYCLES&quot;, &quot;BLENDER_EEVEE&quot;]):
            The rendering engine to use. Either CYCLES or BLENDER_EEVEE.
        render_samples (int, optional):
            Number of samples to render. Defaults to 64.

    Raises:
        ValueError: If the engine is not CYCLES or BLENDER_EEVEE.
    """
    if engine == "CYCLES":
        cycles_init(render_samples)
    elif engine == "BLENDER_EEVEE":
        eevee_init(render_samples)
    else:
        raise ValueError(f"Unknown engine: {engine}")


def eevee_init(render_samples: int):
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.eevee.taa_render_samples = render_samples
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_bloom = True
    bpy.context.scene.render.use_high_quality_normals = True
    bpy.context.preferences.system.memory_cache_limit = 4096
    # Enable GPU compute device for EEVEE
    try:
        prefs = bpy.context.preferences
        prefs.addons["cycles"].preferences.compute_device_type = "CUDA"
        for device in prefs.addons["cycles"].preferences.devices:
            if device.type == "CUDA":
                device.use = True
    except Exception as e:
        print(f"Could not enable GPU: {e}")
    

def cycles_init(render_samples: int):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = render_samples
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.film_transparent = True
    
    # Enable GPU rendering with CUDA/OptiX
    bpy.context.scene.cycles.device = "GPU"
    
    prefs = bpy.context.preferences.addons["cycles"].preferences
    # Try OptiX first (fastest on NVIDIA), fallback to CUDA
    # try:
    #     prefs.compute_device_type = "OPTIX"
    # except:
    #     print(f"Could not enable OptiX: {e}")
    #     prefs.compute_device_type = "CUDA"
    
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type in {"CUDA"}
    
    print(f"Cycles using: {prefs.compute_device_type}")
    for d in prefs.devices:
        if d.use:
            print(f"  - {d.name} ({d.type})")
    bpy.context.scene.render.use_lock_interface = True