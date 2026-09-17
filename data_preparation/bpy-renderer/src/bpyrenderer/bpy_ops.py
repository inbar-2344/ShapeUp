import os
import json
import numpy as np
import sys
from bpyrenderer.camera import add_camera
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_background_color, set_env_map
from bpyrenderer.importer import load_file, load_armature, load_armature_no_animation
from bpyrenderer.render_output import (
    enable_color_output,
    enable_albedo_output,
    enable_depth_output,
    enable_normals_output,
)
from bpyrenderer import SceneManager
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.utils import convert_normal_to_webp, get_keyframes
from os.path import join as pjoin

# Default HDRI environment map, resolved relative to the installed package so the
# repo stays portable (this file lives at bpy-renderer/src/bpyrenderer/bpy_ops.py,
# the asset at bpy-renderer/assets/env_textures/).
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ENV_MAP = os.environ.get(
    "BPYRENDERER_ENV_MAP",
    pjoin(_PACKAGE_ROOT, "assets", "env_textures", "brown_photostudio_02_1k.exr"),
)


class bpy_ops:
    def __init__(self, save_dir):
        # 1. Init engine and scene manager
        init_render_engine("CYCLES")
        self.scene_manager = SceneManager()
        self.scene_manager.clear()
        self.depth_dir = "depth"
        self.image_mv_dir = "mv"
        self.normal_dir = "normal"
        self.save_dir = save_dir
        self.camera_info = None
     
    def count_geometry(self):
        return len(self.scene_manager.data_meshes)   
     
    def count_frames(self):
        return len(self.scene_manager.data_meshes)   
     
    def scene_setting(self):
        # Others. smooth objects and normalize scene
        self.scene_manager.smooth()
        self.scene_manager.clear_normal_map()
        self.scene_manager.set_material_transparency(False)
        self.scene_manager.set_materials_opaque()  # !!! Important for render normal but may cause render error !!!
        self.scene_manager.normalize_scene(1.0)
        # 3. Set environment
        set_env_map(DEFAULT_ENV_MAP)
        self.scene_manager.ensure_gpu()

    # def load_scene(self, glb_path):
    #     self.scene_manager.clear()
    #     self.scene_manager.gc()
    #     load_file(glb_path)
    #     self.scene_setting()
    #     obj_id = os.path.basename(glb_path)[:-4]
    #     self.out_path = os.path.join(self.save_dir, obj_id)
    
    def load_scene(self, glb_path, ignore_components =[]):
        self.scene_manager.clear()
        self.scene_manager.gc()
        load_armature_no_animation(glb_path, ignore_components)
        self.scene_setting()
        obj_id = os.path.basename(glb_path)[:-4]
        self.out_path = os.path.join(self.save_dir, obj_id)
    
    def load_scene_armature(self, glb_path, ignore_components =[]):
        self.scene_manager.clear()
        self.scene_manager.gc()
        load_armature(glb_path, ignore_components)
        self.scene_manager.update_scene_frames()
        self.scene_manager.smooth()
        self.scene_manager.normalize_scene(1.0, process_frames=True, use_parent_node=True)
        self.scene_setting()
        obj_id = os.path.basename(glb_path)[:-4]
        self.out_path = os.path.join(self.save_dir, obj_id)


    def load_single_frame_scene(self, glb_path, frame_number=0):
        self.scene_manager.load_single_frame_scene(glb_path, frame_number)
    
    def get_removal_candidates(self):
        branching_node = self.scene_manager.get_meshes_first_common_parent()
        if not branching_node:
            return None, None
        parts = self.scene_manager.get_geom_first_children(branching_node)
        parts_bbox_volumes = [self.scene_manager.get_obj_bbox_volume(part) for part in parts]
        max_vol = max(parts_bbox_volumes)
        min_index = parts_bbox_volumes.index(max_vol)
        del parts[min_index]
        return parts, parts_bbox_volumes

    def get_most_different_keyframes(self, num_frames=3):
        """Get the most different keyframes from the animation"""
        return self.scene_manager.get_most_different_keyframes(num_frames)


    def get_frame_bbox(self, frame_index):
        """Get the bounding box for a specific frame
        
        Args:
            frame_index: The frame number to get the bounding box for
            
        Returns:
            tuple: (bbox_min, bbox_max, bbox_size) where bbox_size is the max dimension
        """
        # Save current frame
        current_frame = self.scene_manager.get_current_frame()
        current_start = self.scene_manager.get_start_frame()
        current_end = self.scene_manager.get_end_frame()
        
        # Set to the desired frame
        self.scene_manager.set_frame(frame_index)
        
        # Get bounding box for this frame
        bbox_min, bbox_max = self.scene_manager.get_scene_bbox()
        
        # Calculate the max dimension of the bounding box
        bbox_size = max(bbox_max - bbox_min)
        
        # Restore original frame
        self.scene_manager.set_frame(current_frame, current_start, current_end)
        
        return bbox_min, bbox_max, bbox_size
    def remove_part_from_scene(self, part):
        self.scene_manager.delete_hierarchy(part)
    
    def set_camera(self, center = (0, 0, 0), radius = 1.8):
        self._set_cameras(center, radius)
    
    def get_camera(self):
        return self.camera_info
    
    
    def _set_cameras_motion(self):
        # 4. Prepare cameras
        cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.8,
        elevations=[0, 0, 0, 0, 70, -70],
        azimuths=[x - 90 for x in [0, 90, 180, 270, 180, 0]]
        )
        cameras = []
        for i, camera_mat in enumerate(cam_mats):
            camera = add_camera(camera_mat, "ORTHO", add_frame=i < len(cam_mats) - 1)
            cameras.append(camera)
        self.camera_info = {"cam_pos": cam_pos, "cameras": cameras,  "elevations": elevations, "azimuths": azimuths, "cam_mats": cam_mats}
    
    def _set_cameras(self, center = (0, 0, 0), radius = 1.8):
        # randomly sample 20 elevation values between -15 and 30
        # randomly sample 20 azimuth values between -75 and 75
        elevations =np.random.randint(-15, 31, 6).tolist()
        #elevations = np.zeros(20, dtype=int).tolist() # TODO:For guy! change to random after use
        azimuths = np.random.randint(-75, 75, 6).tolist()
        #azimuths = np.round(np.linspace(-70, 70, 20)).astype(int).tolist() # TODO:For guy! change to random after use
        # 4. Prepare cameras
        cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
        center=center,
        radius=radius,
        elevations=[0, 0, 0, 0, 70, -70] + elevations,
        azimuths=[x - 90 for x in [0, 90, 180, 270, 180, 0] + azimuths]
        )
        cameras = []
        for i, camera_mat in enumerate(cam_mats):
            camera = add_camera(camera_mat, "ORTHO", add_frame=i < len(cam_mats) - 1)
            cameras.append(camera)
        self.camera_info = {"cam_pos": cam_pos, "cameras": cameras,  "elevations": elevations, "azimuths": azimuths, "cam_mats": cam_mats}

    def _set_output_dir_for_renders(self, out_dir, mv= True, normal=True, depth=True, width=1024, height=1024):
        if mv:
            enable_color_output(
                width,
                height,
                (pjoin(out_dir, self.image_mv_dir)),
                file_format="WEBP",
                mode="IMAGE",
                film_transparent=True,
            )
        if depth:
            enable_depth_output(pjoin(out_dir, self.depth_dir))
        if normal:
            enable_normals_output(pjoin(out_dir, self.normal_dir))
    
    
    def render(self, part_ind,  mv = True, normal = True, depth = True):
        self.scene_manager.ensure_gpu()
        width, height = 1024, 1024
        out_dir = pjoin(self.out_path,  part_ind)
        self._set_output_dir_for_renders(out_dir, mv, normal, depth, width, height)
        devnull = open(os.devnull, 'w')
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            self.scene_manager.render()
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            devnull.close()
        # Optional. convert normal (.exr) into .webp
        for file in os.listdir(out_dir):
            if file.startswith("normal_") and file.endswith(".exr"):
                filepath = pjoin(out_dir, file)
                render_filepath = filepath.replace("normal_", "render_").replace(
                    ".exr", ".webp"
                )
                convert_normal_to_webp(
                    filepath,
                    filepath.replace(".exr", ".webp"),
                    render_filepath,
                )
                os.remove(filepath)

        # Optional. save metadata
        meta_info = {"width": width, "height": height, "locations": []}
        for i in range(len(self.camera_info["cam_pos"])):
            index = "{0:04d}".format(i)
            meta_info["locations"].append(
                {
                    "index": index,
                    "projection_type": self.camera_info["cameras"][i].data.type,
                    "ortho_scale": self.camera_info["cameras"][i].data.ortho_scale,
                    "camera_angle_x": self.camera_info["cameras"][i].data.angle_x,
                    "elevation": self.camera_info["elevations"][i],
                    "azimuth": self.camera_info["azimuths"][i],
                    "transform_matrix": self.camera_info["cam_mats"][i].tolist(),
                }
            )
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta_info, f, indent=4)

    def get_frames_count(self):
        return self.scene_manager.get_animation_frames()

    def set_frame(self, frame_number):
        self.scene_manager.set_frame(frame_number)

    def export_glb(self, fname="out.glb", export_animations=True, export_current_frame=False):
        save_path = pjoin(self.out_path, fname)
        self.scene_manager.export(save_path, export_animations=export_animations, export_current_frame=export_current_frame)
        return save_path

    def export_current_pose(self, fname="out.glb"):
        save_path = pjoin(self.out_path, fname)
        self.scene_manager.export_current_pose(save_path)
        return save_path

    def gc(self):
        self.scene_manager.gc()

    def copy_frame_to_end(self, frame_number):
        self.scene_manager.copy_frame_to_end(frame_number)