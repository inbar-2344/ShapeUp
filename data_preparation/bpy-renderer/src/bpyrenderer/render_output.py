import math
import os
import json
from typing import List, Optional, Literal, Tuple, Dict

import bpy
import mathutils
import numpy as np
from PIL import Image, ImageDraw
import bpy_extras

from .utils import get_local2world_mat, PRESET_COLORS


def enable_color_output(
    width: int,
    height: int,
    output_dir: Optional[str] = "",
    file_prefix: str = "render_",
    file_format: Literal["WEBP", "PNG"] = "WEBP",
    mode: Literal["IMAGE", "VIDEO"] = "IMAGE",
    **kwargs,
):
    film_transparent = kwargs.get("film_transparent", True)
    fps = kwargs.get("fps", 24)

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = film_transparent
    scene.render.image_settings.quality = 100

    if mode == "IMAGE":
        scene.render.image_settings.file_format = file_format
        scene.render.image_settings.color_mode = "RGBA"
        # scene.render.image_settings.color_depth = "16"
    elif mode == "VIDEO":
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.fps = fps
    scene.render.filepath = os.path.join(output_dir, file_prefix)


def make_normal_to_rgb_node_group(node_tree, editor_type="Compositor"):
    """Create a node group that converts normal vectors to RGB colors for visualization.

    Args:
        node_tree: The node tree to add nodes to
        editor_type: Either "Shader" or "Compositor"

    Returns:
        Tuple of (input_socket, output_socket) for the conversion
    """
    link = lambda from_socket, to_socket: node_tree.links.new(from_socket, to_socket)
    sep_color_node = node_tree.nodes.new(f"{editor_type}NodeSeparateColor")

    def create_normal_to_rgb_map_node():
        node = node_tree.nodes.new(f"{editor_type}NodeMapRange")
        if editor_type == "Shader":
            node.clamp = True  # Clamp
        elif editor_type == "Compositor":
            node.use_clamp = True  # Clamp
        node.inputs["From Min"].default_value = -1.0  # From Min
        node.inputs["From Max"].default_value = 1.0  # From Max
        node.inputs["To Min"].default_value = 0.0  # To Min
        node.inputs["To Max"].default_value = 1.0  # To Max
        return node

    if editor_type == "Shader":
        map_range_node_output_socket_name = "Result"
        converter_io_node_socket_name = "Color"
    elif editor_type == "Compositor":
        map_range_node_output_socket_name = "Value"
        converter_io_node_socket_name = "Image"

    map_range_nodes = {k: create_normal_to_rgb_map_node() for k in ["R", "G", "B"]}
    comb_color_node = node_tree.nodes.new(f"{editor_type}NodeCombineColor")

    link(sep_color_node.outputs["Red"], map_range_nodes["R"].inputs["Value"])
    link(sep_color_node.outputs["Green"], map_range_nodes["G"].inputs["Value"])
    link(sep_color_node.outputs["Blue"], map_range_nodes["B"].inputs["Value"])

    link(
        map_range_nodes["R"].outputs[map_range_node_output_socket_name],
        comb_color_node.inputs["Red"],
    )
    link(
        map_range_nodes["G"].outputs[map_range_node_output_socket_name],
        comb_color_node.inputs["Green"],
    )
    link(
        map_range_nodes["B"].outputs[map_range_node_output_socket_name],
        comb_color_node.inputs["Blue"],
    )

    # return (input socket, output socket)
    return (
        sep_color_node.inputs[converter_io_node_socket_name],
        comb_color_node.outputs[converter_io_node_socket_name],
    )


def set_file_output_non_color(node):
    """Set file output node to use non-color data management."""
    if int(bpy.app.version_string[0]) >= 4:
        node.format.color_management = "OVERRIDE"
        node.format.view_settings.view_transform = "Raw"
    else:
        node.format.color_management = "OVERRIDE"
        node.format.display_settings.display_device = "None"


def enable_normals_output(
    output_dir: Optional[str] = "",
    file_prefix: str = "normal_",
    use_rgb_conversion: bool = True,
    file_format: Literal["OPEN_EXR", "WEBP", "PNG"] = "PNG",
):
    """
    Enable normal output in pure world-space coordinates without any transformations.
    This is the simplest version that directly outputs Blender's world-space normals.

    When use_rgb_conversion=True and file_format in ["WEBP", "PNG"]:
    - World-space normals in range [-1, 1] are mapped to RGB values in range [0, 1]
    - This allows visualization in standard image formats while preserving normal data
    - To recover normals: (rgb_values * 2 - 1) gives world-space normals in [-1, 1] range

    Args:
        output_dir: Output directory for normal files
        file_prefix: Prefix for output files
        use_rgb_conversion: Whether to convert normals to RGB for better visualization
        file_format: Output file format
    """
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.scene.render.use_compositing = True

    # ADDED: Remove old normal output nodes if they exist
    nodes_to_remove = []
    for node in tree.nodes:
        if node.type == 'OUTPUT_FILE' and node.name.startswith("File Output"):
            # This catches unnamed normal output nodes
            nodes_to_remove.append(node)
        elif "NormalOutput" in node.name:
            nodes_to_remove.append(node)
    
    for node in nodes_to_remove:
        tree.nodes.remove(node)

    normal_file_output = tree.nodes.new("CompositorNodeOutputFile")
    normal_file_output.base_path = output_dir
    normal_file_output.location.x = 400
    normal_file_output.file_slots.values()[0].path = file_prefix

    if use_rgb_conversion and file_format in ["WEBP", "PNG"]:
        # Convert normals to RGB for better visualization
        normal_trans_input_socket, normal_trans_output_socket = (
            make_normal_to_rgb_node_group(tree, editor_type="Compositor")
        )

        # Set alpha channel
        set_normal_alpha_node = tree.nodes.new("CompositorNodeSetAlpha")
        set_normal_alpha_node.mode = "REPLACE_ALPHA"

        # Connect nodes
        tree.links.new(rl.outputs["Normal"], normal_trans_input_socket)
        tree.links.new(
            normal_trans_output_socket, set_normal_alpha_node.inputs["Image"]
        )
        tree.links.new(rl.outputs["Alpha"], set_normal_alpha_node.inputs["Alpha"])
        tree.links.new(
            set_normal_alpha_node.outputs["Image"], normal_file_output.inputs["Image"]
        )

        # Configure output format
        if file_format == "WEBP":
            normal_file_output.format.file_format = "WEBP"
            normal_file_output.format.quality = 100
            normal_file_output.format.color_depth = "8"
        elif file_format == "PNG":
            normal_file_output.format.file_format = "PNG"
            normal_file_output.format.color_depth = "16"

        set_file_output_non_color(normal_file_output)

    else:
        # Direct output for EXR or raw normal data
        tree.links.new(rl.outputs["Normal"], normal_file_output.inputs["Image"])

        if file_format == "OPEN_EXR":
            normal_file_output.format.file_format = "OPEN_EXR"
            normal_file_output.format.color_mode = "RGBA"
            normal_file_output.format.color_depth = "32"


def enable_depth_output(output_dir: Optional[str] = "", file_prefix: str = "depth_"):
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    links = tree.links

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_z = True

        # ADDED: Remove old depth output node if it exists
    if "DepthOutput" in tree.nodes:
        tree.nodes.remove(tree.nodes["DepthOutput"])

    depth_output = tree.nodes.new("CompositorNodeOutputFile")
    depth_output.base_path = output_dir
    depth_output.name = "DepthOutput"
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"
    depth_output.file_slots.values()[0].path = file_prefix

    links.new(rl.outputs["Depth"], depth_output.inputs["Image"])


def enable_albedo_output(output_dir: Optional[str] = "", file_prefix: str = "albedo_"):
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_diffuse_color = True

    alpha_albedo = tree.nodes.new(type="CompositorNodeSetAlpha")
    tree.links.new(rl.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    tree.links.new(rl.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.base_path = output_dir
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = "PNG"
    albedo_file_output.format.color_mode = "RGBA"
    albedo_file_output.format.color_depth = "16"
    albedo_file_output.file_slots.values()[0].path = file_prefix

    tree.links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])


def enable_pbr_output(output_dir, attr_name, color_mode="RGBA", file_prefix: str = ""):
    if file_prefix == "":
        file_prefix = attr_name.lower().replace(" ", "-") + "_"

    # Process each material to set up AOV outputs
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue

        node_tree = material.node_tree
        if not node_tree:
            continue

        nodes = node_tree.nodes

        # Check if Principled BSDF node exists
        if "Principled BSDF" not in nodes:
            continue

        principled_node = nodes["Principled BSDF"]

        # Check if the attribute exists in the Principled BSDF node
        if attr_name not in principled_node.inputs:
            print(
                f"Warning: Attribute '{attr_name}' not found in Principled BSDF node for material '{material.name}'"
            )
            continue

        attr_input = principled_node.inputs[attr_name]

        if attr_input.is_linked:
            linked_node = attr_input.links[0].from_node
            linked_socket = attr_input.links[0].from_socket

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.name = attr_name
            node_tree.links.new(linked_socket, aov_output.inputs[0])

        else:
            fixed_value = attr_input.default_value
            if isinstance(fixed_value, float):
                value_node = nodes.new("ShaderNodeValue")
                value_node.outputs[0].default_value = fixed_value
            else:
                value_node = nodes.new("ShaderNodeRGB")
                value_node.outputs[0].default_value = fixed_value

            aov_output = nodes.new("ShaderNodeOutputAOV")
            aov_output.name = attr_name
            node_tree.links.new(value_node.outputs[0], aov_output.inputs[0])

    # Set up compositor nodes
    tree = bpy.context.scene.node_tree
    links = tree.links
    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]

    pbr_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    pbr_file_output.base_path = output_dir
    pbr_file_output.file_slots[0].use_node_format = True
    pbr_file_output.format.file_format = "PNG"
    pbr_file_output.format.color_mode = color_mode
    pbr_file_output.format.color_depth = "16"
    pbr_file_output.file_slots.values()[0].path = file_prefix

    # Add AOV to view layer
    bpy.ops.scene.view_layer_add_aov()
    bpy.context.scene.view_layers["ViewLayer"].active_aov.name = attr_name

    # Set up alpha compositing
    pbr_alpha = tree.nodes.new(type="CompositorNodeSetAlpha")
    tree.links.new(rl.outputs[attr_name], pbr_alpha.inputs["Image"])
    tree.links.new(rl.outputs["Alpha"], pbr_alpha.inputs["Alpha"])

    links.new(pbr_alpha.outputs["Image"], pbr_file_output.inputs["Image"])


def get_keypoint_data(keypoint_names: Optional[List] = None):
    # Set keypoint colors
    keypoint_colors = {}
    if keypoint_names is None:
        for obj in bpy.context.scene.objects:
            if obj.type == "ARMATURE":
                for i, bone in enumerate(obj.pose.bones):
                    keypoint_colors[bone.name.lower().split(":")[-1]] = PRESET_COLORS[
                        i % len(PRESET_COLORS)
                    ]
    elif isinstance(keypoint_names, dict):
        keypoint_colors = keypoint_names
    elif isinstance(keypoint_names, list):
        keypoint_colors = {
            keypoint_names[i]: PRESET_COLORS[i] for i in range(len(keypoint_names))
        }
    else:
        raise ValueError("keypoint_names must be a list or dictionary")

    # Get camera and scene dimensions
    camera = bpy.context.scene.camera
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    # Store keypoint data for all bones
    keypoint_data = {}

    # Process each armature in the scene
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            for bone in obj.pose.bones:
                if not bone.name.lower().split(":")[-1] in keypoint_colors.keys():
                    continue

                # Get bone head in world coordinates
                head_world = obj.matrix_world @ bone.head

                # Project to 2D using Blender's projection
                head_2d = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, camera, head_world
                )

                # Convert to screen coordinates
                head_screen = (int(head_2d.x * width), int((1 - head_2d.y) * height))

                # get color
                keypoint_color = keypoint_colors.get(
                    bone.name.lower().split(":")[-1], list(keypoint_colors.values())[0]
                )

                # Get parent bone name if exists
                parent_name = bone.parent.name if bone.parent else None

                # Store data for this bone
                keypoint_data[bone.name] = {
                    "head_3d": list(head_world),
                    "head_2d": head_screen,
                    "color": keypoint_color,
                    "parent": parent_name,
                }

    return keypoint_data


def visualize_keypoint_map(
    keypoint_data: Dict,
    bone_color: Tuple = (200, 200, 200),
    bone_width: int = 3,
    keypoint_radius: int = 3,
    color_mode: Literal["RGBA", "RGB"] = "RGB",
    background_color: Tuple = (0, 0, 0),
    plot_bones: Optional[List] = None,
):
    """Visualize the skeleton of armature in the scene using only head positions
    and parent-child relationships. Creates a 2D visualization using projected
    coordinates from camera view.
    """
    # Get camera and scene dimensions
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    # Create a new image with transparent background
    if color_mode == "RGB":
        image = Image.new("RGB", (width, height), background_color)
    elif color_mode == "RGBA":
        image = Image.new("RGBA", (width, height), background_color + (0,))
    else:
        raise ValueError(
            f"Unsupported color mode: {color_mode}. Must be either 'RGB' or 'RGBA'."
        )
    draw = ImageDraw.Draw(image)

    # Draw keypoints and connections
    for bone_name, data in list(keypoint_data.items())[::-1]:
        # Check if this bone should be plotted
        plot_bone = (
            bone_name.lower().split(":")[-1] in plot_bones
            if plot_bones is not None
            else True
        )

        if not plot_bone:
            continue

        # Draw connection to parent if exists
        if data["parent"] and data["parent"] in keypoint_data:
            parent_data = keypoint_data[data["parent"]]
            draw.line(
                [data["head_2d"], parent_data["head_2d"]],
                fill=bone_color,
                width=bone_width,
            )

        # Draw keypoint
        draw.ellipse(
            [
                data["head_2d"][0] - keypoint_radius,
                data["head_2d"][1] - keypoint_radius,
                data["head_2d"][0] + keypoint_radius,
                data["head_2d"][1] + keypoint_radius,
            ],
            fill=data["color"],
        )

    return image


def render_keypoint_map(
    output_dir: Optional[str] = "",
    file_prefix: str = "keypoint_",
    file_format: Literal["WEBP", "PNG"] = "WEBP",
    export_meta: bool = True,
    **kwargs,
):
    # kwargs for visualizing the bones
    keypoint_names = kwargs.get("keypoint_names", None)
    bone_color = kwargs.get("bone_color", (200, 200, 200))
    bone_width = kwargs.get("bone_width", 3)
    keypoint_radius = kwargs.get("keypoint_radius", 3)
    background_color = kwargs.get("background_color", (0, 0, 0))
    plot_bones = kwargs.get("plot_bones", None)

    # Save the image
    os.makedirs(output_dir, exist_ok=True)
    file_suffix = ".png" if file_format == "PNG" else ".webp"

    # Initialize metadata
    keypoint_metas = {
        "bone_names": [],
        "head_3d": [],
        "head_2d": [],
        "colors": [],
        "parents": [],
    }

    original_frame = bpy.context.scene.frame_current
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        output_path = os.path.join(output_dir, f"{file_prefix}{frame:04d}{file_suffix}")
        keypoint_data = get_keypoint_data(keypoint_names)
        keypoint_map = visualize_keypoint_map(
            keypoint_data=keypoint_data,
            bone_color=bone_color,
            bone_width=bone_width,
            keypoint_radius=keypoint_radius,
            background_color=background_color,
            plot_bones=plot_bones,
        )
        keypoint_map.save(output_path, file_format, quality=100)

        if export_meta:
            # Initialize frame data
            if frame == bpy.context.scene.frame_start:
                keypoint_metas["bone_names"] = list(keypoint_data.keys())
                keypoint_metas["colors"] = np.array(
                    [data["color"] for data in keypoint_data.values()]
                )
                keypoint_metas["parents"] = [
                    data["parent"] for data in keypoint_data.values()
                ]

            # Collect frame data
            frame_head_3d = np.array(
                [data["head_3d"] for data in keypoint_data.values()]
            )
            frame_head_2d = np.array(
                [data["head_2d"] for data in keypoint_data.values()]
            )

            keypoint_metas["head_3d"].append(frame_head_3d)
            keypoint_metas["head_2d"].append(frame_head_2d)

    if export_meta:
        # Convert lists to numpy arrays
        keypoint_metas["head_3d"] = np.array(keypoint_metas["head_3d"])
        keypoint_metas["head_2d"] = np.array(keypoint_metas["head_2d"])

        # Save metadata
        meta_file = os.path.join(output_dir, f"{file_prefix.split('_')[0]}.npy")
        np.save(meta_file, keypoint_metas)

    bpy.context.scene.frame_set(original_frame)
