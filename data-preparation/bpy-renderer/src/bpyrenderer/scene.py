import bpy
import os
import math
import numpy as np
import bmesh
import bpy
import mathutils
from mathutils import Vector
from typing import Optional, Literal
from .utils import get_keyframes
import gc as python_gc
# Force CUDA (not OPTIX) to avoid shader compilation hangs on headless systems
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'

# Get available devices and enable only CUDA devices (not OPTIX)
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    # Only enable CUDA devices, disable OPTIX to avoid compilation issues
    if device.type == 'CUDA':
        device.use = True
    else:
        device.use = False

# Print which devices are being used for debugging
print("Cycles using: CUDA")
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.use:
        print(f"  - {device.name} ({device.type})")

class SceneManager:
    @property
    def objects(self):
        return bpy.context.scene.objects

    @property
    def scene_meshes(self):
        return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    @property
    def scene_armatures(self):
        return [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]

    @property
    def data_meshes(self):
        return [obj for obj in bpy.data.objects if obj.type == "MESH"]

    @property
    def root_objects(self):
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj

    @property
    def num_frames(self):
        return bpy.context.scene.frame_end + 1

    def ensure_gpu(self):
        # Force settings - use CUDA only (not OPTIX) to avoid compilation hangs
        bpy.context.scene.render.engine = "CYCLES"  # Make sure engine is CYCLES!
        bpy.context.scene.cycles.device = "GPU"
        
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"  # Force CUDA, not OPTIX
        prefs.get_devices()
        
        for device in prefs.devices:
            if device.type == "CUDA":
                device.use = True
            else:
                device.use = False  # Disable OPTIX devices
        

    def get_scene_bbox(self, single_obj=None, ignore_matrix=False):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3

        meshes = self.scene_meshes if single_obj is None else [single_obj]
        if len(meshes) == 0:
            raise RuntimeError("No objects in scene to compute bounding box for")

        for obj in meshes:
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

        return Vector(bbox_min), Vector(bbox_max)

    def get_scene_bbox_all_frames(self):
        """Get bounding box that contains the entire animation sequence"""
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3

        # Store current frame
        current_frame = bpy.context.scene.frame_current

        # Iterate through all frames
        for frame in range(
            bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1
        ):
            bpy.context.scene.frame_set(frame)
            frame_min, frame_max = self.get_scene_bbox()
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, frame_min))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, frame_max))

        # Restore original frame
        bpy.context.scene.frame_set(current_frame)

        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene(
        self,
        normalize_range: float = 1.0,
        range_type: Literal["CUBE", "SPHERE"] = "CUBE",
        process_frames: bool = False,
        use_parent_node: bool = False,
    ):
        # Recompute bounding box, offset and scale
        if process_frames:
            bbox_min, bbox_max = self.get_scene_bbox_all_frames()
        else:
            bbox_min, bbox_max = self.get_scene_bbox()

        if range_type == "CUBE":
            scale = normalize_range / max(bbox_max - bbox_min)
        elif range_type == "SPHERE":
            scale = normalize_range / (bbox_max - bbox_min).length
        else:
            raise ValueError(
                f"Invalid range_type: {range_type}. Must be either 'CUBE' or 'SPHERE'"
            )

        # Calculate offset to center
        offset = -(bbox_min + bbox_max) / 2

        if use_parent_node:
            # Create a new empty object as parent
            parent = bpy.data.objects.new("NormalizationNode", None)
            bpy.context.scene.collection.objects.link(parent)

            # Parent all root objects to the new node
            for obj in self.root_objects:
                if obj is not parent:
                    obj.parent = parent
                    # Keep the object's local transform
                    obj.matrix_parent_inverse = parent.matrix_world.inverted()

            # Set parent's location and scale
            parent.scale = (scale, scale, scale)
            parent.location = offset * scale  # !!!important for use_parent_node!!!
        else:
            # Original behavior: modify each object directly
            for obj in self.root_objects:
                obj.matrix_world.translation += offset
                # Scale relative to world center by adjusting translation and scale
                original_translation = obj.matrix_world.translation.copy()
                obj.matrix_world.translation = original_translation * scale
                obj.scale = obj.scale * scale
                bpy.context.view_layer.update()

        # Restore original frame
        bpy.ops.object.select_all(action="DESELECT")

    def rotate_model(self, object, rotateQuaternion):
        object.select_set(True)
        bpy.context.view_layer.objects.active = object
        object.rotation_mode = "QUATERNION"
        object.rotation_quaternion = mathutils.Quaternion(rotateQuaternion)
        bpy.ops.object.transform_apply()

    def get_most_different_keyframes(self, num_frames=3):
        """Get the most different keyframes from the animation based on actual pose differences"""
        armatures = self.scene_armatures
        keyframes = get_keyframes(armatures)
        # Fall back to object-transform keyframes when there is no armature.
        if len(keyframes) == 0:
            keyframes = get_keyframes(list(bpy.context.scene.objects))
        if len(keyframes) <= num_frames:
            return keyframes

        keyframes = sorted(keyframes)
        
        # Save current frame
        current_frame = bpy.context.scene.frame_current
        
        # Compute feature vectors for each keyframe
        # Features: bbox center (3), bbox size (3), and mesh vertex positions (sampled)
        frame_features = {}
        
        for frame in keyframes:
            bpy.context.scene.frame_set(frame)
            
            # Get bounding box
            try:
                bbox_min, bbox_max = self.get_scene_bbox()
                bbox_center = (bbox_min + bbox_max) / 2
                bbox_size = bbox_max - bbox_min
                
                # Sample vertex positions from all meshes
                vertex_samples = []
                for obj in self.scene_meshes[:5]:  # Limit to first 5 meshes for speed
                    if len(obj.data.vertices) > 0:
                        # Sample up to 50 vertices per mesh
                        step = max(1, len(obj.data.vertices) // 50)
                        for i in range(0, len(obj.data.vertices), step):
                            v = obj.data.vertices[i]
                            world_pos = obj.matrix_world @ v.co
                            vertex_samples.extend([world_pos.x, world_pos.y, world_pos.z])
                
                # Combine features
                feature = list(bbox_center) + list(bbox_size) + vertex_samples
                frame_features[frame] = np.array(feature, dtype=np.float32)
            except Exception as e:
                # If feature extraction fails, use bbox only
                try:
                    bbox_min, bbox_max = self.get_scene_bbox()
                    bbox_center = (bbox_min + bbox_max) / 2
                    bbox_size = bbox_max - bbox_min
                    feature = list(bbox_center) + list(bbox_size)
                    frame_features[frame] = np.array(feature, dtype=np.float32)
                except:
                    # Last resort: just use frame number as feature
                    frame_features[frame] = np.array([float(frame)], dtype=np.float32)
        
        # Restore original frame
        bpy.context.scene.frame_set(current_frame)
        
        # Greedy selection: pick frames that are maximally different
        selected = []
        
        # Start with the frame that has maximum bbox size (usually most interesting)
        max_size_frame = max(keyframes, key=lambda f: np.linalg.norm(frame_features[f][3:6]) if len(frame_features[f]) >= 6 else 0)
        selected.append(max_size_frame)
        
        # Greedily select remaining frames
        while len(selected) < num_frames and len(selected) < len(keyframes):
            best_frame = None
            best_distance = -1
            
            for frame in keyframes:
                if frame in selected:
                    continue
                
                # Compute minimum distance to already selected frames
                min_dist = float('inf')
                for sel_frame in selected:
                    # Use normalized feature vectors for fair comparison
                    feat1 = frame_features[frame]
                    feat2 = frame_features[sel_frame]
                    
                    # Pad shorter feature to match lengths
                    max_len = max(len(feat1), len(feat2))
                    if len(feat1) < max_len:
                        feat1 = np.pad(feat1, (0, max_len - len(feat1)), mode='constant')
                    if len(feat2) < max_len:
                        feat2 = np.pad(feat2, (0, max_len - len(feat2)), mode='constant')
                    
                    dist = np.linalg.norm(feat1 - feat2)
                    min_dist = min(min_dist, dist)
                
                # Select frame with maximum minimum distance (maximize diversity)
                if min_dist > best_distance:
                    best_distance = min_dist
                    best_frame = frame
            
            if best_frame is not None:
                selected.append(best_frame)
            else:
                break
        
        # Sort selected frames by frame number for processing order
        return sorted(selected)

    def render(self):
        """Render only the current frame instead of the entire animation"""
        bpy.context.scene.render.use_compositing = True
        bpy.context.scene.use_nodes = True

        tree = bpy.context.scene.node_tree
        if "Render Layers" not in tree.nodes:
            tree.nodes.new("CompositorNodeRLayers")
        else:
            tree.nodes["Render Layers"]
        bpy.ops.render.render(animation=True, write_still=True)
    
    def smooth(self):
        for obj in self.scene_meshes:
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = np.deg2rad(30)

    def clear_normal_map(self):
        for material in bpy.data.materials:
            material.use_nodes = True
            node_tree = material.node_tree
            try:
                bsdf = node_tree.nodes["Principled BSDF"]
                if bsdf.inputs["Normal"].is_linked:
                    for link in bsdf.inputs["Normal"].links:
                        node_tree.links.remove(link)
            except:
                pass

    def set_material_transparency(self, show_transparent_back: bool) -> None:
        """Set transparency settings for materials with blend mode 'BLEND'.

        Args:
            show_transparent_back: Whether to show the back face of transparent materials.
        """
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue

            if material.blend_method == "BLEND":
                material.show_transparent_back = show_transparent_back

    def set_materials_opaque(self) -> None:
        """Set all materials to opaque blend mode.

        This is useful for rendering passes like normal maps that require
        fully opaque materials for correct results.
        """
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue

            material.blend_method = "OPAQUE"

    def update_scene_frames(
        self, mode: Literal["auto", "manual"] = "auto", num_frames: Optional[int] = None
    ):
        if mode == "auto":
            armatures = self.scene_armatures
            keyframes = get_keyframes(armatures)
            # Fall back to object-transform keyframes when there is no armature
            # (objects animated via location/rotation/scale rather than a rig).
            if len(keyframes) == 0:
                keyframes = get_keyframes(list(bpy.context.scene.objects))
            bpy.context.scene.frame_end = (
                int(max(keyframes)) if len(keyframes) > 0 else 0
            )
        elif mode == "manual":
            if num_frames is None:
                raise ValueError(f"num_frames must be provided if the mode is 'manual'")
            bpy.context.scene.frame_end = num_frames - 1

    def clear(
        self,
        clear_objects: Optional[bool] = True,
        clear_nodes: Optional[bool] = True,
        reset_keyframes: Optional[bool] = True,
        reset_materials: Optional[bool] = True
    ):
        
        if clear_objects:
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            objects = [x for x in bpy.data.objects]
            for obj in objects:
                bpy.data.objects.remove(obj, do_unlink=True)

        # Clear all nodes
        if clear_nodes:
            bpy.context.scene.use_nodes = True
            node_tree = bpy.context.scene.node_tree
            for node in node_tree.nodes:
                node_tree.nodes.remove(node)

        # Reset keyframes
        if reset_keyframes:
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = 0
            for a in bpy.data.actions:
                bpy.data.actions.remove(a)
        
        # Reset material     
        if reset_materials:
            for m in bpy.data.materials:
                bpy.data.materials.remove(m)

    def gc(self):
        python_gc.collect()
        devnull = open(os.devnull, 'w')
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            for _ in range(10):
                bpy.ops.outliner.orphans_purge()
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            devnull.close()

    def get_ancestors(self, obj):
        """Return list of ancestors up to root (including obj)."""
        ancestors = []
        while obj:
            ancestors.append(obj)
            obj = obj.parent
        return ancestors

    def get_meshes_first_common_parent(self):
        """Find the first common ancestor of given objects."""
        objs = self.data_meshes
        if not objs:
            return None

        # Get ancestors for each object
        ancestor_lists = [self.get_ancestors(o) for o in objs]

        # Reverse so root is first
        ancestor_lists = [list(reversed(lst)) for lst in ancestor_lists]

        # Walk until divergence
        common = []
        for zipped in zip(*ancestor_lists):
            if all(z == zipped[0] for z in zipped):
                common.append(zipped[0])
            else:
                break

        return common[-1] if common else None
    
    def delete_hierarchy(self, obj):
        # Clear scene
        # bpy.ops.wm.read_factory_settings(use_empty=True)

        # # Import GLB
        # bpy.ops.import_scene.gltf(filepath=filepath)

        # Find root object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        for child in obj.children_recursive:
            child.select_set(True)
        # Delete them
        bpy.ops.object.delete()

    def has_mesh_descendant(self, obj):
        """Return True if obj itself is a mesh or has any mesh in its subtree."""
        if obj.type == 'MESH':
            return True
        for child in obj.children:
            if self.has_mesh_descendant(child):
                return True
        return False
    
    def get_geom_first_children(self, obj):
        """
        Return a list of first-degree children of obj,
        iff they are meshes themselves or contain meshes in their subtree.
        """
        result = []
        for child in obj.children:
            if self.has_mesh_descendant(child):
                result.append(child)
        return result
    
    def get_object_centroid(self, obj, world_space=True):
        """Return the centroid of a mesh object."""
        if obj.type != 'MESH':
            raise TypeError(f"Object {obj.name} is not a mesh")

        mesh = obj.data
        if not mesh.vertices:
            return None

        total = mathutils.Vector((0.0, 0.0, 0.0))
        for v in mesh.vertices:
            co = v.co
            if world_space:
                co = obj.matrix_world @ co
            total += co

        return total / len(mesh.vertices)

    def get_scene_centroid(self, world_space=True):
        """Return the centroid of all mesh objects in the scene."""
        mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        if not mesh_objs:
            return None

        total = mathutils.Vector((0.0, 0.0, 0.0))
        count = 0

        for obj in mesh_objs:
            c = self.get_object_centroid(obj, world_space=world_space)
            if c:
                total += c
                count += 1

        return total / count if count > 0 else None

    def get_all_meshes_under(self, obj):
        """
        Return all MESH objects under obj (recursively).
        Includes obj itself if it's a mesh.
        """
        meshes = []

        def recurse(o):
            if o.type == 'MESH':
                meshes.append(o)
            for child in o.children:
                recurse(child)

        recurse(obj)
        return meshes
    
    def mean_vector(self, vectors):
        """
        Compute the mean (average) of a list of mathutils.Vector.
        """
        if not vectors:
            return None
        
        total = mathutils.Vector((0.0,) * len(vectors[0]))
        for v in vectors:
            total += v
        return total / len(vectors)


    def distance_from_scene_centroid(self, obj):
        """Return the distance from obj's centroid to the scene centroid."""
        meshes = self.get_all_meshes_under(obj)
        obj_c = self.mean_vector([self.get_object_centroid(mesh, world_space=False) for mesh in meshes])
        scene_c = self.get_scene_centroid(world_space=False)

        if obj_c is None or scene_c is None:
            return None

        return (obj_c - scene_c).length
    
    def get_obj_bbox_volume(self, obj, ignore_matrix=False):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3

        meshes = self.get_all_meshes_under(obj)
        if len(meshes) == 0:
            raise RuntimeError("No objects in scene to compute bounding box for")

        for obj in meshes:
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

        volume = (bbox_max[0] - bbox_min[0]) * (bbox_max[1] - bbox_min[1]) * (bbox_max[2] - bbox_min[2])
        
        return volume
        
    def set_frame(self, frame_number, start_frame=None, end_frame=None):
        """Set the current frame for animation"""
        bpy.context.scene.frame_set(frame_number)
        bpy.context.scene.frame_start = frame_number if start_frame is None else start_frame
        bpy.context.scene.frame_end = frame_number if end_frame is None else end_frame
                    
    def get_animation_frames(self):
        """Get the number of frames in the animation"""
        return bpy.context.scene.frame_end + 1

    def get_current_frame(self):
        """Get the current frame number"""
        return bpy.context.scene.frame_current

    def get_start_frame(self):
        """Get the start frame number"""
        return bpy.context.scene.frame_start

    def get_end_frame(self):
        """Get the end frame number"""
        return bpy.context.scene.frame_end

    def apply_armature_as_shape(self):
        """Apply armature modifiers to get the current frame's deformed mesh"""
        # Store the current frame
        current_frame = bpy.context.scene.frame_current
        
        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')
        
        # Find all mesh objects and apply their armature modifiers
        for obj in self.scene_meshes:
            # Select the mesh object
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            # Apply all armature modifiers
            for modifier in obj.modifiers:
                if modifier.type == 'ARMATURE':
                    try:
                        # Apply the modifier with the current pose
                        bpy.ops.object.modifier_apply(modifier=modifier.name)
                    except:
                        print(f"Could not apply modifier {modifier.name} on {obj.name}")
            
            obj.select_set(False)
        
        # Remove armatures after applying
        for armature in self.scene_armatures:
            bpy.data.objects.remove(armature, do_unlink=True)
        
        bpy.ops.object.select_all(action='DESELECT')
    
    def export_current_pose(self, filepath):
        """Export the current frame's pose using evaluated mesh data"""        
        # Store original selection
        original_selected = [obj for obj in bpy.context.selected_objects]
        original_active = bpy.context.view_layer.objects.active
        
        # Make sure we're in object mode
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Get the dependency graph to access evaluated (deformed) meshes
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')
        
        # Create new mesh objects with deformed geometry
        exported_objects = []
        mesh_data_to_cleanup = []  # Store mesh data references before deleting objects
        for obj in self.scene_meshes:
            # Get the evaluated (deformed) version of the mesh
            obj_eval = obj.evaluated_get(depsgraph)
            
            # Create a new mesh from the evaluated data
            mesh_eval = bpy.data.meshes.new_from_object(obj_eval)
            
            # Create a new object with this mesh
            new_obj = bpy.data.objects.new(obj.name + "_posed", mesh_eval)
            
            # Copy transform from original
            new_obj.matrix_world = obj.matrix_world
            
            # Link to scene and select
            bpy.context.collection.objects.link(new_obj)
            new_obj.select_set(True)
            exported_objects.append(new_obj)
            mesh_data_to_cleanup.append(mesh_eval)  # Store mesh data reference
        
        if not exported_objects:
            raise RuntimeError("No mesh objects found to export")
        
        # Export the posed meshes
        bpy.ops.export_scene.gltf(
            filepath=filepath,
            export_format='GLB',
            use_selection=True
        )
        
        # Clean up: delete the temporary posed objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in exported_objects:
            obj.select_set(True)
        bpy.ops.object.delete()
        
        # Clean up orphaned mesh data (using stored references)
        for mesh_data in mesh_data_to_cleanup:
            if mesh_data and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)
        
        # Restore original selection
        for obj in original_selected:
            if obj.name in bpy.data.objects:
                obj.select_set(True)
        if original_active and original_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = original_active
    # def export_current_pose(self, filepath):
    #     """Export the current frame's pose by duplicating and applying modifiers"""
    #     # Store original selection
    #     original_selected = [obj for obj in bpy.context.selected_objects]
    #     original_active = bpy.context.view_layer.objects.active
        
    #     # Deselect all
    #     bpy.ops.object.select_all(action='DESELECT')
        
    #     # Select all mesh objects
    #     mesh_objects = []
    #     for obj in self.scene_meshes:
    #         obj.select_set(True)
    #         mesh_objects.append(obj)
        
    #     # Duplicate selected objects
    #     bpy.ops.object.duplicate()
    #     duplicated = [obj for obj in bpy.context.selected_objects]
        
    #     # Apply armature modifiers on duplicates
    #     for obj in duplicated:
    #         bpy.context.view_layer.objects.active = obj
    #         for modifier in obj.modifiers:
    #             if modifier.type == 'ARMATURE':
    #                 try:
    #                     bpy.ops.object.modifier_apply(modifier=modifier.name)
    #                 except:
    #                     pass
        
    #     # Deselect originals, keep only duplicates selected
    #     for obj in mesh_objects:
    #         obj.select_set(False)
        
    #     # Export only selected (duplicates)
    #     bpy.ops.export_scene.gltf(
    #         filepath=filepath, 
    #         export_format='GLB',
    #         use_selection=True
    #     )
        
    #     # Delete duplicates
    #     bpy.ops.object.delete()
        
    #     # Restore original selection
    #     for obj in original_selected:
    #         obj.select_set(True)
    #     bpy.context.view_layer.objects.active = original_active

    def export(self, filepath, export_animations=True, export_current_frame=False):
        devnull = open(os.devnull, 'w')
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            bpy.ops.export_scene.gltf(filepath=filepath, export_format='GLB', export_animations=export_animations, export_current_frame=export_current_frame)
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            devnull.close()

    def copy_frame_to_end(self, source_frame):
        """
        Copy all keyframe data from source_frame and append it to the end of the sequence.
        Returns the new frame number where the data was copied to.
        """
        # Get the current end frame
        current_end = bpy.context.scene.frame_end
        target_frame = current_end + 1
        
        # Iterate through all objects (including armatures and meshes)
        for obj in bpy.data.objects:
            if obj.animation_data and obj.animation_data.action:
                action = obj.animation_data.action
                
                # Copy keyframes for all fcurves
                for fcurve in action.fcurves:
                    # Find keyframe at source_frame
                    for keyframe in fcurve.keyframe_points:
                        if int(keyframe.co[0]) == source_frame:
                            # Insert new keyframe at target_frame with same value
                            fcurve.keyframe_points.insert(
                                target_frame, 
                                keyframe.co[1],
                                options={'FAST'}
                            )
                            # Copy interpolation settings
                            new_kf = fcurve.keyframe_points[-1]
                            if keyframe.interpolation in {'CONSTANT', 'LINEAR', 'BEZIER', 'SINE', 'QUAD', 
                                                        'CUBIC', 'QUART', 'QUINT', 'EXPO', 'CIRC', 
                                                        'BACK', 'BOUNCE', 'ELASTIC'}:
                                new_kf.interpolation = keyframe.interpolation
                            else:
                                new_kf.interpolation = 'LINEAR'  # Safe default
                                
                            # Handle types can also be invalid, so protect those too
                            try:
                                new_kf.handle_left_type = keyframe.handle_left_type
                                new_kf.handle_right_type = keyframe.handle_right_type
                            except (TypeError, ValueError):
                                # Use default handle types if copying fails
                                new_kf.handle_left_type = 'AUTO'
                                new_kf.handle_right_type = 'AUTO'
                            break
        
        # Update frame_end
        bpy.context.scene.frame_end = target_frame
        
        return target_frame
        
    
    