import bpy
from mathutils import Euler, Matrix, Vector
from typing import Literal


def init_camera():
    bpy.ops.object.camera_add(location=(0, 0, 0))
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.context.scene.camera = obj

    return bpy.context.scene.camera


def add_camera(
    cam2world_matrix: Matrix,
    camera_type: Literal["PERSP", "ORTHO"] = "PERSP",
    camera_sensor_width: int = 32,
    camera_lens: int = 35,
    ortho_scale: int = 1.1,
    add_frame: bool = False,
):
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)
    if bpy.context.scene.camera is None:
        bpy.ops.object.camera_add(location=(0, 0, 0))
        for obj in bpy.data.objects:
            if obj.type == "CAMERA":
                bpy.context.scene.camera = obj

    cam_ob = bpy.context.scene.camera
    cam_ob.data.type = camera_type
    cam_ob.data.sensor_width = camera_sensor_width
    if camera_type == "PERSP":
        cam_ob.data.lens = camera_lens
    elif camera_type == "ORTHO":
        cam_ob.data.ortho_scale = ortho_scale
    cam_ob.matrix_world = cam2world_matrix

    frame = bpy.context.scene.frame_end
    cam_ob.keyframe_insert(data_path="location", frame=frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=frame)
    cam_ob.data.keyframe_insert(data_path="type", frame=frame)
    cam_ob.data.keyframe_insert(data_path="sensor_width", frame=frame)

    if camera_type == "ORTHO":
        cam_ob.data.keyframe_insert(data_path="ortho_scale", frame=frame)
    elif camera_type == "PERSP":
        cam_ob.data.keyframe_insert(data_path="lens", frame=frame)

    if add_frame:
        bpy.context.scene.frame_end += 1

    return cam_ob
