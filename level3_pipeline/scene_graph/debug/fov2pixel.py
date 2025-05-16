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
    
    # 额外的旋转矩阵，将默认的相机坐标系转换为你的相机坐标系
    adjustment_rotation = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])
    adjustment_rotation = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])
    # 旋转
    print(translated_point)
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
    if camera_point[0] <= 1e-5:
        return -1, -1
    
    x = (camera_point[1] * focal_length) / camera_point[0]
    y = (camera_point[2] * focal_length) / camera_point[0]
    
    # 转换为图像坐标系中的像素位置
    pixel_x = int(image_width / 2 - x)
    pixel_y = int(image_height / 2 - y)
    
    return pixel_x, pixel_y

def world_to_image(world_point, camera_position, camera_rotation, fovx, image_width, image_height):
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
    
    return pixel_x, pixel_y
def world_to_image(world_point, camera_position, camera_rotation, fovx, image_width, image_height):
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
    print(camera_point)
    pixel_x, pixel_y = camera_to_image(camera_point, fovx, image_width, image_height)
    
    return pixel_x, pixel_y

# 示例数据
camera_position = np.array([0.500349, -3.79372, 1.44133])

#world_point = np.array([1,0.2,0.2])

#camera_rotation = R.from_euler('xyz', [0, 0,0], degrees=True).as_matrix()
#Pose([4.1808, -3.95818, 1.73462], [0.506975, 0.493028, 0.502549, 0.497337])
#Pose([4.13347, -0.0873732, 0.575155], [0.499526, 0.499532, 0.500472, 0.500471])

world_point = np.array([4.13347, -0.0873732, 0.575155])
quat =   [0.977196, -0.00524873, 0.0244395, 0.210863]
quat = quat[1:] + quat[:1]
camera_rotation = R.from_quat(quat).as_matrix()
rpy = R.from_matrix(camera_rotation).as_euler('xyz', degrees=True)
print(rpy)
fovx = np.deg2rad(90)
image_width = 1920
image_height = 1080

# 将世界坐标系中的点投影到图像平面
pixel_x, pixel_y = world_to_image(world_point, camera_position, camera_rotation, fovx, image_width, image_height)

# 打印结果
print("Pixel coordinates:", pixel_x, pixel_y)