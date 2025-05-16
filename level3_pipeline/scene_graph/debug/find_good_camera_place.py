import numpy as np
from scipy.spatial.transform import Rotation as R
import sapien
import cv2

def create_and_mount_camera(scene,
                            pose:sapien.Pose,
                            near:float, 
                            far:float, 
                            width:float=1920, 
                            height:float=1080, 
                            fovy:float=np.deg2rad(60),                
                            camera_name:str='camera'):
    camera_mount_actor = scene.create_actor_builder().build_kinematic(name=f'{camera_name}_mount')
    #cannot set fovx for mounted camera
    #after fovy is set, fovx is calculated automatically
    camera = scene.add_mounted_camera(
        name=camera_name,
        mount=camera_mount_actor,
        pose=pose,
        width=width,
        height=height,
        fovy=fovy,
        near=near,
        far=far
    )
    return camera
    pass

def render_image(scene, camera):
    scene.step()
    scene.step()
    scene.update_render()
    camera.take_picture()
    #'get_picture' function requires the color type. 'normal', 'color', 'depth', 'segmentation' etc.
    rgba = camera.get_picture('Color')
    rgba_img = (rgba * 255).clip(0, 255).astype('uint8')
    img = Image.fromarray(rgba_img)
    
    return img

def world_to_image(point, camera_pose, camera_rotation, fovy, width, height):
    """
    将世界坐标系中的点转换到图像坐标系。
    
    参数:
    - point: 世界坐标系中的点 (x, y, z)
    - camera_pose: 相机的位置 (x, y, z)
    - camera_rotation: 相机的旋转矩阵
    - fovy: 相机的视场角（弧度）
    - width: 图像宽度
    - height: 图像高度
    
    返回:
    - 图像坐标系中的点 (u, v)
    """
    # 将点转换到相机坐标系
    point_camera = np.dot(camera_rotation, point - camera_pose)
    
    # 投影到图像平面
    focal_length = height / (2 * np.tan(fovy / 2))
    u = focal_length * point_camera[0] / point_camera[2] + width / 2
    v = focal_length * point_camera[1] / point_camera[2] + height / 2
    
    return np.array([u, v])

def calculate_area_from_image(img, rect_points, camera_pose, camera_rotation, fovy, width, height):
    """
    计算矩形在图像中占据的面积。
    
    参数:
    - img: 渲染的图像
    - rect_points: 矩形的四个顶点坐标 [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
    - camera_pose: 相机的位置 (x, y, z)
    - camera_rotation: 相机的旋转矩阵
    - fovy: 相机的视场角（弧度）
    - width: 图像宽度
    - height: 图像高度
    
    返回:
    - area: 矩形在图像中占据的面积
    """
    # 将矩形的四个顶点转换到图像坐标系
    image_points = [world_to_image(point, camera_pose, camera_rotation, fovy, width, height) for point in rect_points]
    
    # 计算图像边界
    img_rect = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    
    # 计算矩形与图像边界的交集
    intersection = cv2.intersectConvexConvex(np.array(image_points), img_rect)[0]
    
    # 计算交集矩形的面积
    if intersection is not None:
        area = cv2.contourArea(intersection)
    else:
        area = 0
    
    return area

def calculate_rect_area_in_image(scene, camera_pose, camera_angles, fovy, rect_points, width, height):
    """
    计算矩形在图像中占据的面积。
    
    参数:
    - scene: 3D 场景
    - camera_pose: 相机的位置 (x, y, z)
    - camera_angles: 相机的角度 (pitch, yaw)
    - fovy: 相机的视场角（弧度）
    - rect_points: 矩形的四个顶点坐标 [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
    - width: 图像宽度
    - height: 图像高度
    
    返回:
    - area: 矩形在图像中占据的面积
    """
    pitch, yaw = camera_angles
    rpy = [pitch, yaw, 0]
    quat = R.from_euler('xyz', rpy).as_quat()
    camera_rotation = R.from_quat(quat).as_matrix()
    
    # 渲染图像
    camera = create_and_mount_camera(scene,
                                                     pose=sapien.Pose(p=camera_pose, q=quat),
                                                     near=0.1,
                                                     far=1000,
                                                     width=width,
                                                     height=height,
                                                     fovy=fovy,
                                                     camera_name='camera')
    img = render_image(scene, camera)
    
    # 计算矩形在图像中占据的面积
    area = calculate_area_from_image(img, rect_points, camera_pose, camera_rotation, fovy, width, height)
    
    return area

# 示例数据
scene = ...  # 你的 3D 场景
rect_points = [
    np.array([1, 1, 0]),
    np.array([1, -1, 0]),
    np.array([-1, -1, 0]),
    np.array([-1, 1, 0])
]  # 矩形的四个顶点坐标
fixed_x = 0
fixed_y = 0
z_min = 1
z_max = 10
fovy_min = np.deg2rad(30)
fovy_max = np.deg2rad(120)
width = 1366
height = 768

# 优化相机 z 坐标、角度和视场角
best_z, best_pitch, best_yaw, best_fovy = optimize_camera(scene, rect_points, fixed_x, fixed_y, z_min, z_max, fovy_min, fovy_max)

# 打印结果
print("Best z:", best_z)
print("Best pitch:", best_pitch)
print("Best yaw:", best_yaw)
print("Best fovy:", best_fovy)