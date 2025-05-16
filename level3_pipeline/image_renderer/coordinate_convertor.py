import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d
import glog


def world_to_camera(world_point, camera_position, camera_rotation):
    """
    Turn a point in world coordinate system to camera coordinate system.

    Args:
    - world_point: A point in world coordinate system (x, y, z)
    - camera_position: The position of the camera in world coordinate system (x, y, z)
    - camera_rotation: The rotation matrix of the camera (3x3)

    Returns:
    - The point in camera coordinate system (x, y, z)
    """
    # Translate
    translated_point = world_point - camera_position
    # Rotate
    camera_point = np.dot(camera_rotation.T, translated_point)

    return camera_point


def camera_to_image(camera_point, fovx, image_width, image_height):
    """
    Project a point in camera coordinate system to the image plane.

    Args:
    - camera_point: A point in camera coordinate system (x, y, z)
    - fovx: The field of view of the camera (horizontal)
    - image_width: The width of the image (pixels)
    - image_height: The height of the image (pixels)

    Returns:
    - The point on the image plane (pixel_x, pixel_y)

    """
    if camera_point[0] > 0 and camera_point[0] < 1e-6:
        camera_point[0] = 1e-6
    if camera_point[0] < 0 and camera_point[0] > -1e-6:
        camera_point[0] = 1e-6
    # calculate focal length
    focal_length = (image_width / 2) / np.tan(fovx / 2)

    # project to the image plane

    x, y = (camera_point[1] * focal_length) / camera_point[0], (
        camera_point[2] * focal_length
    ) / camera_point[0]

    # convert to pixel coordinates
    pixel_x, pixel_y = int(image_width / 2 - x), int(image_height / 2 - y)

    return pixel_x, pixel_y


def adjust_camera_quaternion_constrained(
    camera_pos, focus_point, original_camera_pos, original_quaternion
):
    """
    根据相机位置变化调整四元数，限制旋转自由度

    Args:
        camera_pos: 新相机位置 [x, y, z]
        focus_point: 焦点位置 [x, y, z]
        original_camera_pos: 原始相机位置 [x, y, z]
        original_quaternion: 原始相机四元数

    Returns:
        quaternion: 调整后的相机四元数
    """
    # 将输入转换为numpy数组
    camera_pos = np.array(camera_pos)
    focus_point = np.array(focus_point)
    original_camera_pos = np.array(original_camera_pos)

    # 计算位置变化
    delta_pos = camera_pos - original_camera_pos

    # 获取原始的欧拉角
    original_euler = transforms3d.euler.quat2euler(original_quaternion, "sxyz")
    roll, pitch, yaw = original_euler

    # 根据位置变化调整角度
    if abs(delta_pos[2]) > 1e-6:  # z轴变化
        # 只调整pitch
        forward = focus_point - camera_pos
        pitch = -np.arctan2(forward[2], np.sqrt(forward[0] ** 2 + forward[1] ** 2))
        # 保持原始的roll和yaw
    elif abs(delta_pos[0]) > 1e-6 or abs(delta_pos[1]) > 1e-6:  # x或y轴变化
        # 调整pitch和yaw
        forward = focus_point - camera_pos
        pitch = -np.arctan2(forward[2], np.sqrt(forward[0] ** 2 + forward[1] ** 2))
        yaw = np.arctan2(forward[1], forward[0])
        # 保持原始的roll

    # 转换回四元数
    new_quaternion = transforms3d.euler.euler2quat(roll, pitch, yaw, "sxyz")

    return new_quaternion


def world_to_image(
    world_point,
    camera_position,
    camera_rotation,
    fovx,
    image_width,
    image_height,
    bound=0,
):
    """
    Turn a point in world coordinate system to the image plane.

    Args:
    - world_point: A point in world coordinate system (x, y, z)
    - camera_position: The position of the camera in world coordinate system (x, y, z)
    - camera_rotation: The rotation matrix of the camera (3x3)
    - fovx: The field of view of the camera (horizontal)
    - image_width: The width of the image (pixels)
    - image_height: The height of the image (pixels)

    Returns:
    - The point on the image plane (pixel_x, pixel_y)

    """
    #  glog.info(f'world_point: {world_point}, camera_position: {camera_position}, camera_rotation: {camera_rotation}, fovx: {fovx}, image_width: {image_width}, image_height: {image_height}, bound: {bound}')

    camera_point = world_to_camera(world_point, camera_position, camera_rotation)
    pixel_x, pixel_y = camera_to_image(camera_point, fovx, image_width, image_height)

    # if pixel_x < -image_width * bound:
    #     pixel_x = -image_width * bound
    # if pixel_x >= image_width + image_width * bound:
    #     pixel_x = image_width + image_width * bound - 1
    # if pixel_y < -image_height * bound:
    #     pixel_y = -image_height * bound
    # if pixel_y >= image_height + image_height * bound:
    #     pixel_y = image_height + image_height * bound - 1

    return np.array([pixel_x, pixel_y])


import numpy as np
import transforms3d


def world_rectangle_to_image_polygon(
    rect_points, camera_pos, camera_rotation, fovx, width, height
):
    """
    将世界坐标系中的矩形转换为图像平面上的多边形，处理视锥体裁剪和边界情况

    Args:
        rect_points: 世界坐标系中矩形的顶点列表 shape=(n,3)
        camera_pos: 相机位置
        camera_rotation: 相机旋转矩阵
        fovx: 水平视场角
        width: 图像宽度
        height: 图像高度

    Returns:
        image_polygon: 图像平面上的多边形顶点列表，如果矩形完全不可见则返回None
    """

    def is_point_behind_camera(point_camera):
        """检查点是否在相机后方"""
        return point_camera[2] <= 0

    def interpolate_point(p1, p2, z):
        """在z平面上插值计算交点"""
        t = (z - p1[2]) / (p2[2] - p1[2])
        return p1 + t * (p2 - p1)

    # 1. 转换到相机坐标系
    points_camera = []
    for point in rect_points:
        # 转换到相机坐标系
        p_rel = point - camera_pos
        p_camera = camera_rotation.T @ p_rel
        p_camera = np.array([p_camera[1], p_camera[2], p_camera[0]])  # 交换x和z轴
        points_camera.append(p_camera)

    points_camera = np.array(points_camera)

    # 2. 处理相机后方的点
    visible_points = []
    n_points = len(points_camera)

    for i in range(n_points):
        p1 = points_camera[i]
        p2 = points_camera[(i + 1) % n_points]

        p1_behind = is_point_behind_camera(p1)
        p2_behind = is_point_behind_camera(p2)

        if not p1_behind:
            visible_points.append(p1)

        # 如果一个点在相机后方，另一个点在前方，计算交点
        if p1_behind != p2_behind:
            intersection = interpolate_point(
                p1, p2, 0.1
            )  # 使用一个小的正值避免数值问题
            visible_points.append(intersection)

    if not visible_points:
        return None

    # 3. 投影到图像平面
    image_points = []
    aspect_ratio = width / height
    fovy = 2 * np.arctan(np.tan(fovx / 2) / aspect_ratio)

    for point in visible_points:
        point = np.array([point[2], point[0], point[1]])
        pixel_x, pixel_y = camera_to_image(point, fovx, width, height)

        image_points.append([pixel_x, pixel_y])

    if len(image_points) < 3:
        return None

    return np.array(image_points)
