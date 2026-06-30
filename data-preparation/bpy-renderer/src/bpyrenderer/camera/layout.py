import math
from typing import List, Tuple, Optional

import numpy as np
from mathutils import Euler, Matrix, Vector


def build_transformation_mat(translation, rotation) -> np.ndarray:
    """Build a transformation matrix from translation and rotation parts.

    :param translation: A (3,) vector representing the translation part.
    :param rotation: A 3x3 rotation matrix or Euler angles of shape (3,).
    :return: The 4x4 transformation matrix.
    """
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(
            f"Translation has invalid shape: {translation.shape}. Must be (3,) or (3,1) vector."
        )
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(Euler(rotation).to_matrix())
    else:
        raise RuntimeError(
            f"Rotation has invalid shape: {rotation.shape}. Must be rotation matrix of shape "
            f"(3,3) or Euler angles of shape (3,) or (3,1)."
        )

    return mat


def get_camera_positions_on_sphere(
    center: Tuple[float, float, float],
    radius: float,
    elevations: List[float],
    num_camera_per_layer: Optional[int] = None,
    azimuth_offset: Optional[float] = 0.0,
    azimuths: Optional[List[float]] = None,
) -> Tuple[List, List, List, List]:
    """
    Get camera positions on a sphere around a center point.

    Places cameras at specified elevation angles around a sphere, with a given number of cameras
    per elevation layer. The cameras are positioned to look at the center point.

    Args:
        center: (x,y,z) coordinates of the sphere center
        radius: Radius of the sphere
        elevations: List of elevation angles in degrees
        num_camera_per_layer: Number of cameras to place at each elevation angle

    Returns:
        Tuple containing:
        - points: List of camera position vectors
        - mats: List of camera transformation matrices
        - elevation_t: List of elevation angles for each camera
        - azimuth_t: List of azimuth angles for each camera
    """
    points, mats, elevation_t, azimuth_t = [], [], [], []

    elevation_deg = elevations
    elevation = np.deg2rad(elevation_deg)

    if num_camera_per_layer is not None and azimuths is None:
        azimuth_deg = np.linspace(0, 360, num_camera_per_layer + 1)[:-1]
        azimuth_deg = azimuth_deg % 360
        if azimuth_offset is not None:
            azimuth_deg += azimuth_offset
    else:
        azimuth_deg = azimuths
    azimuth = np.deg2rad(azimuth_deg)
    for _phi, theta in zip(elevation, azimuth):
        phi = 0.5 * math.pi - _phi
        elevation_t.append(_phi)
        azimuth_t.append(theta)

        r = radius
        x = center[0] + r * math.sin(phi) * math.cos(theta)
        y = center[1] + r * math.sin(phi) * math.sin(theta)
        z = center[2] + r * math.cos(phi)
        cam_pos = Vector((x, y, z))
        points.append(cam_pos)

        center = Vector(center)
        rotation_euler = (center - cam_pos).to_track_quat("-Z", "Y").to_euler()
        cam_matrix = build_transformation_mat(cam_pos, rotation_euler)
        mats.append(cam_matrix)

    return points, mats, elevation_t, azimuth_t

