import bpy
from typing import List
from .custom_loaders import import_vertex_colored_models
from ..utils import get_keyframes


def load_file(path):
    """A naive function"""
    if path.endswith(".vrm"):
        bpy.ops.import_scene.vrm(filepath=path)
    elif path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
    elif path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    elif path.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise RuntimeError(f"Invalid input file: {path}")


def load_armature(path, ignore_components: List = []):
    currentObjects = set(bpy.data.objects)
    load_file(path)
    toRemove = [
        x
        for x in bpy.data.objects
        if x not in currentObjects
        and any([component in x.name for component in ignore_components])
    ]
    for obj in toRemove:
        bpy.data.objects.remove(obj, do_unlink=True)
    objects = [
        x for x in bpy.data.objects if x not in currentObjects and x.type == "ARMATURE"
    ]
    # Some animated glbs drive motion through object-transform keyframes rather than
    # a rig, so there may be no armature. Tolerate that instead of raising IndexError;
    # the loaded scene is still usable for object-transform animation.
    armature = objects[0] if objects else None

    return armature

def load_armature_no_animation(path, ignore_components: List = []):
    currentObjects = set(bpy.data.objects)
    load_file(path)
    toRemove = [
        x
        for x in bpy.data.objects
        if x not in currentObjects
        and any([component in x.name for component in ignore_components])
    ]
    for obj in toRemove:
        bpy.data.objects.remove(obj, do_unlink=True)
