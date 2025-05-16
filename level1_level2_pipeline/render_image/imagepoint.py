import numpy as np
from scipy.spatial.transform import Rotation as R


def world_to_camera(world_point, camera_position, camera_rotation):
    """
    将世界坐标系中的点转换到相机坐标系。

    参数:
    - world_point: 世界坐标系中的点 (x, y, z)
    - camera_position: 相机在世界坐标系中的位置 (x, y, z)
    - camera_rotation: 相机的旋转矩阵 (3x3)

    返回:
    - 相机坐标系中的点 (x, y, z)
    """
    # 平移
    translated_point = world_point - camera_position

    adjustment_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # 旋转
    camera_point = np.dot(camera_rotation.T, translated_point)
    camera_point = np.dot(np.linalg.inv(adjustment_rotation), camera_point)

    return camera_point


def camera_to_image(camera_point, fovx, image_width, image_height):
    """
    将相机坐标系中的点投影到图像平面。

    参数:
    - camera_point: 相机坐标系中的点 (x, y, z)
    - fovx: 相机的视场角（水平）
    - image_width: 图像的宽度（像素）
    - image_height: 图像的高度（像素）

    返回:
    - 图像平面上的点 (pixel_x, pixel_y)
    """
    # 计算焦距
    focal_length = (image_width / 2) / np.tan(fovx / 2)

    # 投影到图像平面
    if camera_point[0] <= 1e-3:
        return -1e6, -1e6

    x = (camera_point[1] * focal_length) / camera_point[0]
    y = (camera_point[2] * focal_length) / camera_point[0]

    # 转换为图像坐标系中的像素位置
    pixel_x = int(image_width / 2 - x)
    pixel_y = int(image_height / 2 - y)

    return pixel_x, pixel_y


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
    将世界坐标系中的点投影到图像平面。

    参数:
    - world_point: 世界坐标系中的点 (x, y, z)
    - camera_position: 相机在世界坐标系中的位置 (x, y, z)
    - camera_rotation: 相机的旋转矩阵 (3x3)
    - fovx: 相机的视场角（水平）
    - image_width: 图像的宽度（像素）
    - image_height: 图像的高度（像素）

    返回:
    - 图像平面上的点 (pixel_x, pixel_y)
    """
    camera_point = world_to_camera(world_point, camera_position, camera_rotation)
    pixel_x, pixel_y = camera_to_image(camera_point, fovx, image_width, image_height)

    if (
        pixel_x < -image_width * bound
        or pixel_x >= image_width + image_width * bound
        or pixel_y < -image_height * bound
        or pixel_y >= image_height + image_height * bound
    ):
        return np.array([-1e6, -1e6])

    return np.array([pixel_x, pixel_y])


def find_valid_pitch_yaw(
    required_points, camera_pose, fovy, width, height, must_top_view=False
):
    """
    找到一组合适的俯仰角（pitch）和偏航角（yaw），使得四个点在相机的视野范围内。

    参数:
    - rect_points: 矩形的四个顶点坐标 [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
    - camera_pose: 相机的位置 (x, y, z)
    - fov: 相机的视场角（弧度）
    - width: 图像宽度
    - height: 图像高度

    返回:
    - pitch: 合适的俯仰角（弧度）
    - yaw: 合适的偏航角（弧度）
    """
    fovx = np.arctan(np.tan(fovy / 2) * width / height) * 2

    # 计算相机到每个点的方向向量
    direction_vectors = [point - camera_pose for point in required_points]

    # 计算平均方向向量
    average_direction = np.mean(direction_vectors, axis=0)

    # 计算yaw值
    yaw = np.arctan2(average_direction[1], average_direction[0])

    pitch_values = np.linspace(-np.pi, np.pi / 10, 72)
    yaw_values = np.linspace(-np.pi / 2 + yaw, np.pi / 2 + yaw, 36)
    if must_top_view:
        pitch_values = np.linspace(-np.pi / 2 - np.pi / 10, -np.pi / 2 + np.pi / 10, 36)

    import random

    random.shuffle(pitch_values)
    random.shuffle(yaw_values)

    for pitch in pitch_values:
        for yaw in yaw_values:
            rpy = [0, pitch, yaw]
            camera_rotation = R.from_euler("xyz", rpy).as_matrix()
            all_in_fov = True
            for point in required_points:
                point_camera = world_to_camera(point, camera_pose, camera_rotation)
                pixel_x, pixel_y = camera_to_image(point_camera, fovx, width, height)
                if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
                    all_in_fov = False
                    break
            if all_in_fov:
                return pitch, yaw
    return None, None
