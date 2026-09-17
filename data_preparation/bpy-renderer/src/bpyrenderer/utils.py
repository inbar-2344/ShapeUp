import numpy as np
from PIL import Image
from typing import List, Tuple
import imageio


def get_local2world_mat(blender_obj) -> np.ndarray:
    """Returns the pose of the object in the form of a local2world matrix.
    :return: The 4x4 local2world matrix.
    """
    obj = blender_obj
    # Start with local2parent matrix (if obj has no parent, that equals local2world)
    matrix_world = obj.matrix_basis

    # Go up the scene graph along all parents
    while obj.parent is not None:
        # Add transformation to parent frame
        matrix_world = (
            obj.parent.matrix_basis @ obj.matrix_parent_inverse @ matrix_world
        )
        obj = obj.parent

    return np.array(matrix_world)


def rgba_to_rgb(rgba_image, bg_color=[255, 255, 255]):
    background = np.array(bg_color)

    # Separate the foreground and alpha
    foreground = rgba_image[..., :3]
    alpha = rgba_image[..., 3:]

    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float) / 255

    rgb_image = alpha * foreground + (1 - alpha) * background
    return rgb_image.astype(np.uint8)


# def get_keyframes(obj_list):
#     keyframes = []
#     for obj in obj_list:
#         anim = obj.animation_data
#         if anim is not None and anim.action is not None:
#             for fcu in anim.action.fcurves:
#                 for keyframe in fcu.keyframe_points:
#                     x, y = keyframe.co
#                     if x not in keyframes:
#                         keyframes.append(int(x))
#     return keyframes

def get_keyframes(obj_list):
    keyframes = set()
    for obj in obj_list:
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            for fcu in anim.action.fcurves:
                for keyframe in fcu.keyframe_points:
                    keyframes.add(int(keyframe.co[0]))
    return sorted(list(keyframes))


def load_image(file_path: str, num_channels: int = 3) -> np.ndarray:
    """Load the image at the given path returns its pixels as a numpy array.

    The alpha channel is neglected.

    :param file_path: The path to the image.
    :param num_channels: Number of channels to return.
    :return: The numpy array
    """
    file_ending = file_path[file_path.rfind(".") + 1 :].lower()
    if file_ending in ["exr", "png", "webp"]:
        return imageio.imread(file_path)[:, :, :num_channels]
    elif file_ending in ["jpg"]:
        import cv2

        img = cv2.imread(file_path)  # reads an image in the BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise NotImplementedError(
            "File with ending " + file_ending + " cannot be loaded."
        )


def convert_normal_to_webp(src: str, dst: str, src_render: str):
    data = load_image(src, 4)
    normal_map = data[:, :, :3] * 255
    try:
        alpha_channel = load_image(src_render, 4)[:, :, 3]
        for i in range(alpha_channel.shape[0]):
            for j in range(alpha_channel.shape[1]):
                alpha_channel[i][j] = 255 if alpha_channel[i][j] > 0 else 0
        normal_map = np.concatenate(
            (normal_map, alpha_channel[:, :, np.newaxis]), axis=2
        )
    except:
        pass

    save_type = dst.split(".")[-1]
    Image.fromarray(normal_map.astype(np.uint8)).save(dst, save_type, quality=100)


def convert_depth_to_webp(src: List[str], dst: List[str]) -> Tuple[float, float]:
    """Convert depth EXR images to PNG format with normalization.

    Args:
        src: List of input EXR image paths
        dst: List of output PNG image paths

    Returns:
        Tuple[float, float]: (min_depth, scale) - The minimum depth value and scale factor used for normalization
    """
    # Read all depth images
    depth_images = []
    valid_masks = []
    min_depth = float("inf")
    max_depth = float("-inf")

    for path in src:
        # Read EXR image
        depth = imageio.imread(path)
        # Create mask for valid depth values
        mask = np.ones_like(depth, dtype=float)
        mask[depth > 1000.0] = 0.0
        depth[~(mask > 0.5)] = 0.0

        # Update min and max depth values
        valid_depths = depth[mask > 0.5]
        if len(valid_depths) > 0:
            min_depth = min(min_depth, valid_depths.min())
            max_depth = max(max_depth, valid_depths.max())

        depth_images.append(depth)
        valid_masks.append(mask)

    # Calculate scale factor for normalization
    scale = 255.0 / (max_depth - min_depth) if max_depth > min_depth else 1.0

    # Process and save each image
    for depth, mask, output_path in zip(depth_images, valid_masks, dst):
        # Normalize depth values
        normalized_depth = (depth - min_depth) * scale
        # Apply mask
        normalized_depth[~(mask > 0.5)] = 0.0
        # Convert to uint8
        depth_uint8 = normalized_depth.astype(np.uint8)
        # Save as PNG
        imageio.imwrite(output_path, depth_uint8)

    return min_depth, scale


PRESET_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (255, 0, 128),  # Pink
    (128, 255, 0),  # Lime
    (0, 255, 128),  # Spring Green
    (0, 128, 255),  # Sky Blue
    (255, 128, 128),  # Light Red
    (128, 192, 64),  # Apple Green
    (128, 128, 255),  # Light Blue
    (255, 255, 128),  # Light Yellow
    (255, 128, 255),  # Light Magenta
    (128, 255, 255),  # Light Cyan
    (192, 64, 0),  # Dark Orange
    (64, 0, 192),  # Dark Purple
    (192, 0, 64),  # Dark Pink
    (64, 192, 0),  # Dark Lime
    (0, 192, 64),  # Dark Spring Green
    (0, 64, 192),  # Dark Sky Blue
    (255, 64, 64),  # Coral
    (64, 255, 64),  # Bright Green
    (64, 64, 255),  # Royal Blue
    (255, 192, 64),  # Gold
    (192, 64, 255),  # Violet
    (64, 255, 192),  # Aquamarine
    (192, 192, 64),  # Olive
    (192, 64, 192),  # Plum
    (64, 192, 192),  # Teal
    (128, 64, 0),  # Brown
    (0, 64, 128),  # Navy
    (128, 0, 64),  # Burgundy
    (64, 128, 0),  # Forest Green
    (0, 128, 64),  # Sea Green
    (64, 0, 128),  # Indigo
    (255, 128, 64),  # Light Orange
    (128, 64, 255),  # Light Purple
    (255, 64, 128),  # Hot Pink
    (128, 255, 64),  # Yellow Green
    (64, 255, 128),  # Mint
    (64, 128, 255),  # Cornflower Blue
    (192, 128, 64),  # Bronze
    (128, 64, 192),  # Amethyst
    (192, 64, 128),  # Rose
    (128, 255, 128),  # Light Green #
    (64, 192, 128),  # Turquoise
    (255, 96, 0),  # Deep Orange
    (96, 0, 255),  # Deep Purple
    (255, 0, 96),  # Deep Pink
    (96, 255, 0),  # Deep Lime
    (0, 255, 96),  # Deep Spring Green
    (0, 96, 255),  # Deep Sky Blue
    (255, 96, 96),  # Deep Coral
    (96, 255, 96),  # Deep Bright Green
    (96, 96, 255),  # Deep Royal Blue
    (255, 224, 96),  # Deep Gold
    (224, 96, 255),  # Deep Violet
    (96, 255, 224),  # Deep Aquamarine
    (224, 224, 96),  # Deep Olive
    (224, 96, 224),  # Deep Plum
    (96, 224, 224),  # Deep Teal
    (160, 96, 0),  # Deep Brown
    (0, 96, 160),  # Deep Navy
    (160, 0, 96),  # Deep Burgundy
    (96, 160, 0),  # Deep Forest Green
    (0, 160, 96),  # Deep Sea Green
    (96, 0, 160),  # Deep Indigo
    (255, 160, 96),  # Deep Light Orange
    (160, 96, 255),  # Deep Light Purple
    (255, 96, 160),  # Deep Hot Pink
    (160, 255, 96),  # Deep Yellow Green
    (96, 255, 160),  # Deep Mint
    (96, 160, 255),  # Deep Cornflower Blue
    (224, 160, 96),  # Deep Bronze
    (160, 96, 224),  # Deep Amethyst
    (224, 96, 160),  # Deep Rose
    (160, 224, 96),  # Deep Apple Green
    (96, 224, 160),  # Deep Turquoise
    (255, 32, 0),  # Bright Red
    (0, 255, 32),  # Bright Green
    (0, 32, 255),  # Bright Blue
    (255, 255, 32),  # Bright Yellow
    (255, 32, 255),  # Bright Magenta
    (32, 255, 255),  # Bright Cyan
    (255, 160, 32),  # Bright Orange
    (160, 32, 255),  # Bright Purple
    (255, 32, 160),  # Bright Pink
    (160, 255, 32),  # Bright Lime
    (32, 255, 160),  # Bright Spring Green
    (32, 160, 255),  # Bright Sky Blue
    (255, 160, 160),  # Bright Light Red
    (160, 255, 160),  # Bright Light Green
    (160, 160, 255),  # Bright Light Blue
    (255, 255, 160),  # Bright Light Yellow
    (255, 160, 255),  # Bright Light Magenta
    (160, 255, 255),  # Bright Light Cyan
]

MIXAMO_KEYPOINTS = [
    "hips",
    "spine",
    "spine1",
    "spine2",
    "rightshoulder",
    "rightarm",
    "rightforearm",
    "righthand",
    "neck",
    "head",
    "leftshoulder",
    "leftarm",
    "leftforearm",
    "lefthand",
    "leftupleg",
    "leftleg",
    "leftfoot",
    "lefttoebase",
    "rightupleg",
    "rightleg",
    "rightfoot",
    "righttoebase",
]


VROID_KEYPOINTS = [  # Note that vroid has not only one template
    "j_bip_c_hips",
    "j_bip_c_spine",
    "j_bip_c_chest",
    "j_bip_c_upperchest",
    "j_bip_c_neck",
    "j_bip_c_head",
    "j_bip_l_shoulder",
    "j_bip_l_upperarm",
    "j_bip_l_lowerarm",
    "j_bip_l_hand",
    "j_bip_r_shoulder",
    "j_bip_r_upperarm",
    "j_bip_r_lowerarm",
    "j_bip_r_hand",
    "j_bip_l_upperleg",
    "j_bip_l_lowerleg",
    "j_bip_l_foot",
    "j_bip_l_toebase",
    "j_bip_r_upperleg",
    "j_bip_r_lowerleg",
    "j_bip_r_foot",
    "j_bip_r_toebase",
]

VROID_KEYPOINTS_2 = [
    "hips",
    "left leg",
    "left knee",
    "left ankle",
    "left toe",
    "right leg",
    "right knee",
    "right ankle",
    "right toe",
    "spine",
    "chest",
    "left shoulder",
    "left arm",
    "left elbow",
    "left wrist",
    "right shoulder",
    "right arm",
    "right elbow",
    "right wrist",
    "neck",
    "head",
]

VROID_KEYPOINTS_3 = [
    "hips",
    "leftupleg",
    "leftleg",
    "leftfoot",
    "lefttoebase",
    "rightupleg",
    "rightleg",
    "rightfoot",
    "righttoebase",
    "spine",
    "chest",
    "spine1",
    "leftshoulder",
    "leftarm",
    "leftforearm",
    "lefthand",
    "neck",
    "head",
    "rightshoulder",
    "rightarm",
    "rightforearm",
    "righthand",
]

VROID_KEYPOINTS_4 = [
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "shoulder_l",
    "upperarm_l",
    "lowerarm_l",
    "left hand",
    "shoulder_r",
    "upperarm_r",
    "lowerarm_r",
    "right hand",
    "upperleg_l",
    "lowerleg_l",
    "foot_l",
    "upperleg_r",
    "lowerleg_r",
    "foot_r",
]

VROID_KEYPOINTS_5 = [
    "hips",
    "spine",
    "chest",
    "upper_chest",
    "neck",
    "head",
    "shoulder.l",
    "upper_arm.l",
    "forearm.l",
    "hand.l",
    "shoulder.r",
    "upper_arm.r",
    "forearm.r",
    "hand.r",
    "thigh.l",
    "shin.l",
    "foot.l",
    "toe.l",
    "thigh.r",
    "shin.r",
    "foot.r",
    "toe.r",
]

VROID_KEYPOINTS_6 = [
    "hips",
    "spine",
    "spine1",
    "spine2",
    "neck",
    "head",
    "shoulder_l",
    "arm_l",
    "forearm_l",
    "hand_l",
    "shoulder_r",
    "arm_r",
    "forearm_r",
    "hand_r",
    "upleg_l",
    "leg_l",
    "foot_l",
    "toe_l",
    "upleg_r",
    "leg_r",
    "foot_r",
    "toe_r",
]

VROID_KEYPOINT_MAPS = {
    "j_bip_l_upperarm": VROID_KEYPOINTS,
    "left arm": VROID_KEYPOINTS_2,
    "leftarm": VROID_KEYPOINTS_3,
    "upperarm_l": VROID_KEYPOINTS_4,
    "upper_arm.l": VROID_KEYPOINTS_5,
    "arm_l": VROID_KEYPOINTS_6,
}
