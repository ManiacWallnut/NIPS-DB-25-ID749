import sapien as sapien
import numpy as np
import transforms3d
from shapely.geometry import Polygon, Point
from scipy.spatial.transform import Rotation as R
from . import image_render_processor, coordinate_convertor
from custom_geometry_helper_new import basic_geometries
from PIL import Image, ImageColor, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2
import logging
import glog


def create_and_mount_camera(
    scene,
    pose: sapien.Pose,
    near: float,
    far: float,
    width: float = 1920,
    height: float = 1080,
    fovy: float = np.deg2rad(60),
    camera_name: str = "camera",
):

    camera_mount_actor = scene.create_actor_builder().build_kinematic(
        name=f"{camera_name}_mount"
    )
    # cannot set fovx for mounted camera
    # after fovy is set, fovx is calculated automatically
    camera = scene.add_mounted_camera(
        name=camera_name,
        mount=camera_mount_actor,
        pose=pose,
        width=width,
        height=height,
        fovy=fovy,
        near=near,
        far=far,
    )
    return camera


def render_image(scene, camera):

    scene.step()
    scene.update_render()
    camera.take_picture()
    rgba = camera.get_picture("Color")
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(rgba_img)

    return img


def box_on_image(height, width, box):
    """使用线段裁剪的改进版box_on_image函数"""
    if len(box) < 4:
        return []

    # 裁剪边界线段并获取可见部分
    visible_edges = []
    for i in range(4):
        p1, p2 = box[i], box[(i + 1) % 4]
        clipped_line, is_visible = clip_line_to_screen(p1, p2, width, height)
        if is_visible:
            visible_edges.append(clipped_line)

    if not visible_edges:  # 如果没有可见边，返回空
        return []

    # 使用可见边界构建新的多边形
    polygon_points = []
    for edge in visible_edges:
        if not polygon_points or not np.array_equal(polygon_points[-1], edge[0]):
            polygon_points.append(edge[0])
        polygon_points.append(edge[1])

    # 确保多边形闭合
    if len(polygon_points) > 2 and not np.array_equal(
        polygon_points[0], polygon_points[-1]
    ):
        polygon_points.append(polygon_points[0])

    # 获取填充点
    points = []
    if len(polygon_points) >= 3:  # 至少需要3个点形成有效多边形
        # 计算边界框
        xs = [p[0] for p in polygon_points]
        ys = [p[1] for p in polygon_points]
        min_x = max(0, int(min(xs)))
        max_x = min(width - 1, int(max(xs)))
        min_y = max(0, int(min(ys)))
        max_y = min(height - 1, int(max(ys)))

        # 使用polygon对象进行内部点测试
        polygon = Polygon(polygon_points)
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if polygon.contains(Point(x, y)):
                    points.append((x, y))

    return points


def free_space_boxes_on_image(height, width, free_spaces):

    front_points, back_points, left_points, right_points = [], [], [], []
    rear_left_points, rear_right_points, front_left_points, front_right_points = (
        [],
        [],
        [],
        [],
    )

    if "front" in free_spaces:
        front_points = box_on_image(height, width, free_spaces["front"])
    if "rear" in free_spaces:
        back_points = box_on_image(height, width, free_spaces["rear"])
    if "left" in free_spaces:
        left_points = box_on_image(height, width, free_spaces["left"])
    if "right" in free_spaces:
        right_points = box_on_image(height, width, free_spaces["right"])
    if "rear-left" in free_spaces:
        rear_left_points = box_on_image(height, width, free_spaces["rear-left"])
    if "rear-right" in free_spaces:
        rear_right_points = box_on_image(height, width, free_spaces["rear-right"])
    if "front-left" in free_spaces:
        front_left_points = box_on_image(height, width, free_spaces["front-left"])
    if "front-right" in free_spaces:
        front_right_points = box_on_image(height, width, free_spaces["front-right"])

    return (
        front_points,
        back_points,
        left_points,
        right_points,
        rear_left_points,
        rear_right_points,
        front_left_points,
        front_right_points,
    )


def draw_non_ground_object_views_on_image(
    scene,
    object_top_center,
    camera_pose,
    camera_rpy,
    free_spaces=[],
    save_path: str = "image.png",
    width=1080,
    height=1080,
    fovy=np.deg2rad(90),
):

    quat = transforms3d.euler.euler2quat(
        camera_rpy[0], camera_rpy[1], camera_rpy[2], axes="sxyz"
    )

    camera = create_and_mount_camera(
        scene,
        pose=sapien.Pose(p=camera_pose, q=quat),
        near=0.1,
        far=1000,
        width=width,
        height=height,
        fovy=fovy,
        camera_name="camera",
    )

    img = render_image(scene, camera)

    circled_numbers = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

    circled_numbers_id = 0

    free_space_points = {}
    EIGHT_DIRECTIONS = [
        "rear",
        "rear-left",
        "left",
        "front-left",
        "front",
        "front-right",
        "right",
        "rear-right",
    ]
    for dir in range(8):
        free_space_points[EIGHT_DIRECTIONS[dir]] = [
            coordinate_convertor.world_to_image(
                np.append(free_spaces[dir][j], object_top_center[2]),
                camera.get_global_pose().p,
                transforms3d.quaternions.quat2mat(camera.get_global_pose().q),
                camera.fovx,
                width,
                height,
            )
            for j in range(4)
        ]
    # print('side',free_space_points)
    front, rear, left, right, rear_left, rear_right, front_left, front_right = (
        free_space_boxes_on_image(
            height=height, width=width, free_spaces=free_space_points
        )
    )
    free_space_points = [
        rear,
        rear_left,
        left,
        front_left,
        front,
        front_right,
        right,
        rear_right,
    ]
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "cyan"]
    # print('all',free_space_points)
    font = ImageFont.truetype("msmincho.ttc", 35)

    camera_rotation_matrix = transforms3d.quaternions.quat2mat(
        camera.get_global_pose().q
    )

    for dir in range(8):
        mid = np.append(
            (free_spaces[dir][0] + free_spaces[dir][2]) * 0.5, object_top_center[2]
        )
        mid = coordinate_convertor.world_to_image(
            mid,
            camera.get_global_pose().p,
            camera_rotation_matrix,
            camera.fovx,
            width,
            height,
        )
        # print('freespace',free_spaces[i], front, rear, left, right)
        for point in free_space_points[dir]:
            current_color = img.getpixel(point)
            new_color = tuple(
                [
                    current_color[i] // 2 + ImageColor.getrgb(colors[dir])[i] // 2
                    for i in range(3)
                ]
            )
            img.putpixel(point, new_color)
        draw = ImageDraw.Draw(img)
        draw.text(mid, circled_numbers[circled_numbers_id], font=font, fill=colors[dir])
        circled_numbers_id += 1

    img.save(save_path)
    camera.disable()


def draw_object_views_on_image(
    scene,
    object_top_center,
    object_bottom=0,
    object_orientation=(1, 0),
    save_path: str = "image.png",
    draw_four_directions=False,
    draw_free_space=False,
    free_spaces=[],
    width=1080,
    height=1080,
    fovy=np.deg2rad(90),
):

    # free_spaces = copy.deepcopy(input_free_spaces)

    top_center = object_top_center
    possible_camera_pose = [top_center + np.array([0, 0, 1])]
    possible_camera_rpy = [
        np.deg2rad(90),
        np.deg2rad(-90),
        -np.arctan2(object_orientation[1], object_orientation[0]),
    ]
    possible_camera_quaternion = transforms3d.euler.euler2quat(
        possible_camera_rpy[0],
        possible_camera_rpy[1],
        possible_camera_rpy[2],
        axes="sxyz",
    )
    possible_camera_quaternion = [
        [
            possible_camera_quaternion[1],
            possible_camera_quaternion[2],
            possible_camera_quaternion[3],
            possible_camera_quaternion[0],
        ]
    ]

    for i in range(4):
        free_space_center = np.append(
            1.25 * free_spaces[i * 2][i]
            + 0.5 * free_spaces[i * 2][(i + 3) % 4]
            - 0.75 * free_spaces[i * 2][(i + 1) % 4],
            top_center[2],
        ) + np.array([0, 0, 1])
        possible_camera_pose.append(free_space_center)
        rpy = basic_geometries.Basic3Dgeometry.vector_to_rpy(
            top_center - free_space_center
        )
        #  rpy[2] = np.arctan2(object_orientation[1], object_orientation[0])

        quat = transforms3d.euler.euler2quat(
            -rpy[0] + np.deg2rad(180), rpy[1], rpy[2], axes="sxyz"
        )
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])

        possible_camera_quaternion.append(quat)

    img_name = ["top_view", "back_view", "left_view", "front_view", "right_view"]

    if draw_four_directions == False:
        img_name = img_name[0:1]
        possible_camera_pose = possible_camera_pose[0:1]
        possible_camera_quaternion = possible_camera_quaternion[0:1]

    for i in range(len(img_name)):
        camera = create_and_mount_camera(
            scene,
            pose=sapien.Pose(
                p=possible_camera_pose[i], q=possible_camera_quaternion[i]
            ),
            near=0.1,
            far=1000,
            width=width,
            height=height,
            fovy=fovy,
            camera_name="camera",
        )

        img = render_image(scene, camera)

        min_distance_for_robot = 0.3
        max_distance_for_display = 0.5

        if draw_free_space:
            free_space_to_camera = {}
            camera_rotation_matrix = transforms3d.quaternions.quat2mat(
                camera.get_global_pose().q
            )

            short_edge_front = np.linalg.norm(free_spaces[4][0] - free_spaces[4][1])
            short_edge_rear = np.linalg.norm(free_spaces[0][0] - free_spaces[0][1])
            short_edge_left = np.linalg.norm(free_spaces[2][0] - free_spaces[2][3])
            short_edge_right = np.linalg.norm(free_spaces[6][0] - free_spaces[6][3])

            if short_edge_front > max_distance_for_display:
                free_spaces[4][1] = (
                    free_spaces[4][0]
                    + (free_spaces[4][1] - free_spaces[4][0])
                    * max_distance_for_display
                    / short_edge_front
                )
                free_spaces[4][2] = (
                    free_spaces[4][3]
                    + (free_spaces[4][2] - free_spaces[4][3])
                    * max_distance_for_display
                    / short_edge_front
                )

            if short_edge_rear > max_distance_for_display:
                free_spaces[0][0] = (
                    free_spaces[0][1]
                    + (free_spaces[0][0] - free_spaces[0][1])
                    * max_distance_for_display
                    / short_edge_rear
                )
                free_spaces[0][3] = (
                    free_spaces[0][2]
                    + (free_spaces[0][3] - free_spaces[0][2])
                    * max_distance_for_display
                    / short_edge_rear
                )
            if short_edge_left > max_distance_for_display:
                free_spaces[2][0] = (
                    free_spaces[2][3]
                    + (free_spaces[2][0] - free_spaces[2][3])
                    * max_distance_for_display
                    / short_edge_left
                )
                free_spaces[2][1] = (
                    free_spaces[2][2]
                    + (free_spaces[2][1] - free_spaces[2][2])
                    * max_distance_for_display
                    / short_edge_left
                )
            if short_edge_right > max_distance_for_display:
                free_spaces[6][2] = (
                    free_spaces[6][1]
                    + (free_spaces[6][2] - free_spaces[6][1])
                    * max_distance_for_display
                    / short_edge_right
                )
                free_spaces[6][3] = (
                    free_spaces[6][0]
                    + (free_spaces[6][3] - free_spaces[6][0])
                    * max_distance_for_display
                    / short_edge_right
                )

            if short_edge_front > min_distance_for_robot:
                free_space_to_camera["front"] = [
                    coordinate_convertor.world_to_image(
                        np.append(free_spaces[4][j], object_top_center[2]),
                        camera.get_global_pose().p,
                        camera_rotation_matrix,
                        camera.fovx,
                        width,
                        height,
                    )
                    for j in range(4)
                ]
            if short_edge_rear > min_distance_for_robot:
                free_space_to_camera["rear"] = [
                    coordinate_convertor.world_to_image(
                        np.append(free_spaces[0][j], object_top_center[2]),
                        camera.get_global_pose().p,
                        camera_rotation_matrix,
                        camera.fovx,
                        width,
                        height,
                    )
                    for j in range(4)
                ]
            if short_edge_left > min_distance_for_robot:
                free_space_to_camera["left"] = [
                    coordinate_convertor.world_to_image(
                        np.append(free_spaces[2][j], object_top_center[2]),
                        camera.get_global_pose().p,
                        camera_rotation_matrix,
                        camera.fovx,
                        width,
                        height,
                    )
                    for j in range(4)
                ]
            if short_edge_right > min_distance_for_robot:
                free_space_to_camera["right"] = [
                    coordinate_convertor.world_to_image(
                        np.append(free_spaces[6][j], object_top_center[2]),
                        camera.get_global_pose().p,
                        camera_rotation_matrix,
                        camera.fovx,
                        width,
                        height,
                    )
                    for j in range(4)
                ]

            free_space_to_camera["rear-left"] = [
                coordinate_convertor.world_to_image(
                    np.append(free_spaces[1][j], object_top_center[2]),
                    camera.get_global_pose().p,
                    camera_rotation_matrix,
                    camera.fovx,
                    width,
                    height,
                )
                for j in range(4)
            ]
            free_space_to_camera["front-left"] = [
                coordinate_convertor.world_to_image(
                    np.append(free_spaces[3][j], object_top_center[2]),
                    camera.get_global_pose().p,
                    camera_rotation_matrix,
                    camera.fovx,
                    width,
                    height,
                )
                for j in range(4)
            ]
            free_space_to_camera["front-right"] = [
                coordinate_convertor.world_to_image(
                    np.append(free_spaces[5][j], object_top_center[2]),
                    camera.get_global_pose().p,
                    camera_rotation_matrix,
                    camera.fovx,
                    width,
                    height,
                )
                for j in range(4)
            ]
            free_space_to_camera["rear-right"] = [
                coordinate_convertor.world_to_image(
                    np.append(free_spaces[7][j], object_top_center[2]),
                    camera.get_global_pose().p,
                    camera_rotation_matrix,
                    camera.fovx,
                    width,
                    height,
                )
                for j in range(4)
            ]

            if "table" in save_path:

                print(f"{save_path[:-4]}_{img_name[i]}.png")

            circled_numbers = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

            circled_numbers_id = 0

            front, back, left, right, rear_left, rear_right, front_left, front_right = (
                free_space_boxes_on_image(height, width, free_space_to_camera)
            )

            EIGHT_DIRECTIONS = [
                "rear",
                "rear-left",
                "left",
                "front-left",
                "front",
                "front-right",
                "right",
                "rear-right",
            ]

            mark_number = []

            if short_edge_front >= min_distance_for_robot:
                mid = np.append(
                    (free_spaces[4][0] + free_spaces[4][2]) * 0.5, object_top_center[2]
                )
                mid = coordinate_convertor.world_to_image(
                    mid,
                    camera.get_global_pose().p,
                    camera_rotation_matrix,
                    camera.fovx,
                    width,
                    height,
                )
                font = ImageFont.truetype("msmincho.ttc", 35)
                for point in front:
                    current_color = img.getpixel(point)
                    new_color = tuple(
                        [
                            current_color[i] // 3 * 2 + ImageColor.getrgb("red")[i] // 3
                            for i in range(3)
                        ]
                    )
                    img.putpixel(point, new_color)
                draw = ImageDraw.Draw(img)
                draw.text(
                    mid, circled_numbers[circled_numbers_id], font=font, fill="black"
                )
                circled_numbers_id += 1
                mark_number.append(EIGHT_DIRECTIONS.index("front"))

            if short_edge_rear >= min_distance_for_robot:
                mid = np.append(
                    (free_spaces[0][0] + free_spaces[0][2]) * 0.5, object_top_center[2]
                )
                mid = coordinate_convertor.world_to_image(
                    mid,
                    camera.get_global_pose().p,
                    camera_rotation_matrix,
                    camera.fovx,
                    width,
                    height,
                )
                font = ImageFont.truetype("msmincho.ttc", 35)
                for point in back:
                    current_color = img.getpixel(point)
                    new_color = tuple(
                        [
                            current_color[i] // 3 * 2
                            + ImageColor.getrgb("yellow")[i] // 3
                            for i in range(3)
                        ]
                    )
                    img.putpixel(point, new_color)
                draw = ImageDraw.Draw(img)
                draw.text(
                    mid, circled_numbers[circled_numbers_id], font=font, fill="black"
                )
                circled_numbers_id += 1
                mark_number.append(EIGHT_DIRECTIONS.index("rear"))

            if short_edge_left >= min_distance_for_robot:
                mid = np.append(
                    (free_spaces[2][0] + free_spaces[2][2]) * 0.5, object_top_center[2]
                )
                mid = coordinate_convertor.world_to_image(
                    mid,
                    camera.get_global_pose().p,
                    camera_rotation_matrix,
                    camera.fovx,
                    width,
                    height,
                )
                font = ImageFont.truetype("msmincho.ttc", 35)
                for point in left:
                    current_color = img.getpixel(point)
                    new_color = tuple(
                        [
                            current_color[i] // 3 * 2
                            + ImageColor.getrgb("blue")[i] // 3
                            for i in range(3)
                        ]
                    )
                    img.putpixel(point, new_color)
                draw = ImageDraw.Draw(img)
                draw.text(
                    mid, circled_numbers[circled_numbers_id], font=font, fill="black"
                )
                circled_numbers_id += 1
                mark_number.append(EIGHT_DIRECTIONS.index("left"))

            if short_edge_right >= min_distance_for_robot:
                mid = np.append(
                    (free_spaces[6][0] + free_spaces[6][2]) * 0.5, object_top_center[2]
                )
                mid = coordinate_convertor.world_to_image(
                    mid,
                    camera.get_global_pose().p,
                    camera_rotation_matrix,
                    camera.fovx,
                    width,
                    height,
                )
                font = ImageFont.truetype("msmincho.ttc", 35)
                for point in right:
                    current_color = img.getpixel(point)
                    new_color = tuple(
                        [
                            current_color[i] // 3 * 2
                            + ImageColor.getrgb("green")[i] // 3
                            for i in range(3)
                        ]
                    )
                    img.putpixel(point, new_color)
                draw = ImageDraw.Draw(img)
                draw.text(
                    mid, circled_numbers[circled_numbers_id], font=font, fill="black"
                )
                circled_numbers_id += 1
                mark_number.append(EIGHT_DIRECTIONS.index("right"))

        img.save(f"{save_path[:-4]}_{img_name[i]}.png")
        camera.disable()

    # img.save(save_path)
    return img, mark_number

    pass


def solve_camera_pose_for_4_points(
    points,
    camera_xyz,
    width,
    height,
    focus_ratio,
    fovy_range=[np.deg2rad(5), np.deg2rad(75)],
    view="human_full",
):

    point_center = np.mean(points, axis=0)

    def calculate_projection_error_top(params):
        fovy, yaw = params
        camera_pose = np.array([camera_xyz[0], camera_xyz[1], camera_xyz[2]])
        camera_rotation = transforms3d.euler.euler2mat(
            0, -np.deg2rad(90), yaw, axes="sxyz"
        )

        target_points = [
            np.array([-1 / 2, 1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([1 / 2, 1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([-1 / 2, -1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([1 / 2, -1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
        ]

        fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)

        image_points = [
            coordinate_convertor.world_to_image(
                points[i],
                camera_pose,
                camera_rotation,
                fovx,
                width,
                height,
                bound=1.0 / 9,
            )
            for i in range(4)
        ]
        image_points /= np.array([width, height])

        total_error = np.sum(
            [np.linalg.norm(image_points[i] - target_points[i]) for i in range(4)]
        )

        return total_error

    def calculate_projection_error(params):
        fovy, roll, pitch = params
        camera_pose = np.array([camera_xyz[0], camera_xyz[1], camera_xyz[2]])

        fixed_yaw = np.arctan2(
            point_center[1] - camera_xyz[1], point_center[0] - camera_xyz[0]
        )

        camera_rotation = transforms3d.euler.euler2mat(
            roll, pitch, fixed_yaw, axes="sxyz"
        )

        target_points = [
            np.array([-1 / 2, 1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([1 / 2, 1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([-1 / 2, -1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
            np.array([1 / 2, -1 / 2]) * focus_ratio + np.array([1 / 2, 1 / 2]),
        ]

        fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)

        image_points = [
            coordinate_convertor.world_to_image(
                points[i],
                camera_pose,
                camera_rotation,
                fovx,
                width,
                height,
                bound=1.0 / 9,
            )
            for i in range(4)
        ]
        image_points /= np.array([width, height])

        total_error = np.sum(
            [np.linalg.norm(image_points[i] - target_points[i]) for i in range(4)]
        )

        return total_error

    if "human" in view:
        roll_init = np.pi / 6
        pitch_init = 0
        fovy_init = (fovy_range[0] + fovy_range[1]) / 2

        bounds = [
            (fovy_range[0], fovy_range[1]),
            (-np.pi / 900, np.pi / 900),
            (0, np.pi / 2),
        ]
        result = minimize(
            calculate_projection_error,
            x0=[fovy_init, roll_init, pitch_init],
            method="COBYLA",
            bounds=bounds,
            options={"ftol": 1e-3, "maxiter": 250},
        )
        total_error = calculate_projection_error(result.x)
        return result.x, total_error
    else:
        fovy_init = (fovy_range[0] + fovy_range[1]) / 2
        yaw_init = 0
        bounds = [
            (fovy_range[0], fovy_range[1]),
            (0, np.pi * 2 - 1e-6),
        ]

        result = minimize(
            calculate_projection_error_top,
            x0=[fovy_init, yaw_init],
            method="COBYLA",
            bounds=bounds,
            options={"ftol": 1e-3, "maxiter": 250},
        )
        total_error = calculate_projection_error_top(result.x)
        return result.x, total_error


# In the future, this function may be moved to scene_graph.py
def auto_get_optimal_camera_pose_for_object(
    view="top_full",  # 'top_focus', 'top_full', 'human_full', 'human_focus'
    camera_xy=[0, 0],
    z_range=[0, 2.5],
    object_bbox=None,
    platform_rect=None,
    width=1920,
    height=1080,
    fovy_range=[np.deg2rad(5), np.deg2rad(100)],
    focus_ratio=0.5,
):

    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

    # logger.info(f"View: {view}")
    # logger.info(f"Camera XY: {camera_xy}")
    # logger.info(f"Z Range: {z_range}")
    # logger.info(f"Object Bounding Box: {object_bbox}")
    # logger.info(f"Platform Rectangle: {platform_rect}")
    # logger.info(f"Width: {width}")
    # logger.info(f"Height: {height}")
    # logger.info(f"Focus Ratio: {focus_ratio}")

    key_points = None
    if "top_focus" in view:
        key_points = [object_bbox[0], object_bbox[3], object_bbox[1], object_bbox[2]]
    elif view == "human_full" or view == "top_full":
        key_points = [
            platform_rect[0],
            platform_rect[3],
            platform_rect[1],
            platform_rect[2],
        ]
    elif view == "human_focus":
        object_mid_line = (object_bbox[1][:2] + object_bbox[2][:2]) / 2 - (
            object_bbox[0][:2] + object_bbox[3][:2]
        ) / 2
        camera_to_object_line = (
            object_bbox[0][:2] + object_bbox[3][:2]
        ) / 2 - camera_xy
        if np.cross(camera_to_object_line, object_mid_line) > 0:
            key_points = [
                object_bbox[1],
                object_bbox[3],
                object_bbox[5],
                object_bbox[7],
            ]
        else:
            key_points = [
                object_bbox[0],
                object_bbox[2],
                object_bbox[4],
                object_bbox[6],
            ]
    else:
        raise ValueError("Invalid view type")

    optimal_error = 1e9
    optimal_z, optimal_roll, optimal_pitch, optimal_fovy = None, None, None, None
    optimal_yaw = None

    if "human" in view:
        for z in np.linspace(z_range[0], z_range[1], 10):
            camera_xyz = [camera_xy[0], camera_xy[1], z]
            [fovy, roll, pitch], error = solve_camera_pose_for_4_points(
                key_points,
                camera_xyz,
                width,
                height,
                focus_ratio,
                fovy_range=fovy_range,
                view=view,
            )
            if error < optimal_error:
                optimal_error = error
                optimal_z, optimal_roll, optimal_pitch, optimal_fovy = (
                    z,
                    roll,
                    pitch,
                    fovy,
                )

        key_point_center = np.mean(key_points, axis=0)
        fixed_yaw = np.arctan2(
            key_point_center[1] - camera_xy[1], key_point_center[0] - camera_xy[0]
        )
        optimal_yaw = fixed_yaw
    else:

        for z in np.linspace(z_range[0], z_range[1], 10):
            camera_xyz = [camera_xy[0], camera_xy[1], z]
            [fovy, yaw], error = solve_camera_pose_for_4_points(
                key_points,
                camera_xyz,
                width,
                height,
                focus_ratio,
                fovy_range=fovy_range,
                view=view,
            )
            if error < optimal_error:
                optimal_error = error
                optimal_z, optimal_roll, optimal_pitch, optimal_fovy = (
                    z,
                    0,
                    np.deg2rad(90),
                    fovy,
                )
                optimal_yaw = yaw
        pass

    fovy_32 = 2 * np.arctan(np.tan(optimal_fovy / 2) * 2)

    optimal_pose = sapien.Pose(
        p=[camera_xy[0], camera_xy[1], optimal_z],
        q=transforms3d.euler.euler2quat(
            optimal_roll, optimal_pitch, optimal_yaw, axes="sxyz"
        ),
    )

    # glog.info(
    #     f"Optimal roll: {optimal_roll}, Optimal pitch: {optimal_pitch}, Optimal yaw: {optimal_yaw}"
    # )
    # print(
    #     "Optimal camera pose:",
    #     optimal_pose,
    #     "Optimal fovy:",
    #     np.rad2deg(optimal_fovy),
    #     "Optimal err:",
    #     optimal_error,
    # )
    return optimal_pose, optimal_fovy


def draw_cuboid_on_image(
    img: Image.Image,
    cuboid_image_points,
    height: int,
    width: int,
    color,
    rectangle_grey: bool = False,
):
    """Draws a cuboid on the image using vectorized operations."""

    # 1. Convert to NumPy array for faster manipulation
    img_array = np.array(img)

    def fill_parallellogram(image, vertices, color):

        img_array = np.array(image)

        color = (color[2], color[1], color[0])
        if len(img_array.shape) == 2:
            # Convert grayscale to BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            # Convert RGBA to BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            # Convert RGB to BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        pts = vertices.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(
            img_array,
            [pts],
            isClosed=True,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # cv2.fillPoly(img_array, [pts], color)

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGBA)

        return img_array

    # 3. Define rectangles (faces) of the cuboid
    rect_in_cube_list = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]

    # 4. Draw each rectangle
    for rect_in_cube in rect_in_cube_list:
        rect_points = [cuboid_image_points[i] for i in rect_in_cube]
        img_array = (
            img_array // 2
            + fill_parallellogram(img_array, np.array(rect_points), color) // 2
        ).astype(np.uint8)
    #        img_array = fill_rectangle(rect_points, img_array, color, rectangle_grey)

    # 5. Convert back to PIL Image
    return Image.fromarray(img_array)


def clip_polygon_to_screen(points, width, height):
    """使用Sutherland-Hodgman算法裁剪多边形"""

    def clip_against_line(poly_points, x1, y1, x2, y2):
        if poly_points is None:
            return []
        """对一条边界线进行裁剪"""
        result = []
        for i in range(len(poly_points)):
            p1 = poly_points[i]
            p2 = poly_points[(i + 1) % len(poly_points)]

            # 计算点在边界的哪一侧
            pos1 = (x2 - x1) * (p1[1] - y1) - (y2 - y1) * (p1[0] - x1)
            pos2 = (x2 - x1) * (p2[1] - y1) - (y2 - y1) * (p2[0] - x1)

            # 两点都在内部
            if pos1 >= 0 and pos2 >= 0:
                result.append(p2)
            # 第一个点在外部，第二个点在内部
            elif pos1 < 0 and pos2 >= 0:
                intersection = compute_intersection(p1, p2, [x1, y1], [x2, y2])
                result.append(intersection)
                result.append(p2)
            # 第一个点在内部，第二个点在外部
            elif pos1 >= 0 and pos2 < 0:
                intersection = compute_intersection(p1, p2, [x1, y1], [x2, y2])
                result.append(intersection)

        return result

    def compute_intersection(p1, p2, p3, p4):
        """计算两条线段的交点"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-8:
            return p1

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]

    # 依次对四个边界进行裁剪
    clipped = points
    clipped = clip_against_line(clipped, 0, 0, width, 0)  # 上边界
    if not clipped:
        return []
    clipped = clip_against_line(clipped, width, 0, width, height)  # 右边界
    if not clipped:
        return []
    clipped = clip_against_line(clipped, width, height, 0, height)  # 下边界
    if not clipped:
        return []
    clipped = clip_against_line(clipped, 0, height, 0, 0)  # 左边界

    return clipped


def draw_freespace_on_image(img, points, color, width, height, draw=True):
    """使用cv2绘制裁剪后的自由空间"""
    # 转换为OpenCV格式
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    # 裁剪多边形
    clipped_points = clip_polygon_to_screen(points, width, height)
    if not clipped_points:
        return img

    # 创建遮罩
    mask = np.zeros_like(img_array)

    # 转换点为整数坐标
    points_array = np.array(clipped_points, dtype=np.int32)
    points_array = points_array.reshape((-1, 1, 2))

    # 填充多边形
    if draw:
        cv2.fillPoly(mask, [points_array], color[::-1])  # BGR顺序

    # 混合原图和填充
    img_array = cv2.addWeighted(img_array, 1, mask, 0.5, 0)

    # 转换回PIL格式
    if img.mode == "RGBA":
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGBA)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_array)


def get_high_contrast_colors(random_seed=2025):
    """返回一组高对比度的深色，适合标注和引导线"""
    deep_colors = [
        # 深红色系
        (180, 30, 45),  # 深红
        (220, 50, 50),  # 鲜红
        (150, 30, 70),  # 酒红
        # 蓝色系
        (30, 70, 150),  # 深蓝
        (0, 90, 170),  # 皇家蓝
        (70, 130, 180),  # 钢蓝
        # 绿色系
        (30, 120, 50),  # 深绿
        (0, 100, 80),  # 森林绿
        (60, 140, 60),  # 草绿
        # 黄/橙色系
        (200, 120, 0),  # 深橙
        (180, 140, 20),  # 金黄
        (215, 150, 0),  # 琥珀
        # 紫色系
        (110, 40, 120),  # 深紫
        (140, 70, 140),  # 紫罗兰
        (90, 50, 140),  # 蓝紫
        # 青色系
        (0, 110, 120),  # 深青
        (40, 140, 140),  # 蓝绿
        (0, 130, 130),  # 墨绿
        # 中性色调
        (80, 80, 80),  # 深灰
        (120, 100, 80),  # 棕褐
        (70, 90, 100),  # 蓝灰
        # 其他高对比度颜色
        (180, 80, 80),  # 印度红
        (70, 100, 50),  # 橄榄绿
        (90, 60, 140),  # 深紫蓝
        (170, 70, 0),  # 赭石
        (80, 60, 30),  # 深棕
        (150, 80, 100),  # 浆果色
    ]
    import random

    random.seed(random_seed)
    random.shuffle(deep_colors)  # 随机打乱颜色顺序

    # 可以根据需要添加更多颜色
    return deep_colors


def auto_render_image_refactored(
    scene,
    name="some_item",
    transparent_item_list=[],
    pose=None,
    fovy=np.deg2rad(75),
    width=1920,
    height=1080,
    might_mark_object_cuboid_list=[],
    might_mark_freespace_list=[],
    rectangle_grey=False,
    save_path="image.png",
    trans_visiblity=0.2,
):
    """
    Auto render image for a scene with specified parameters.

    Args:
        scene: The scene to render.
        name: The name of the item to render. Not necessarily used.
        transparent_item_list: List of items to be rendered with transparency.
        pose: The pose of the camera.
        fovy: The field of view in y-axis in radians.
        width: The width of the rendered image.
        height: The height of the rendered image.
        might_mark_object_cuboid_list: List of cuboids to be marked in the image. Currently we only draw a bounding box around the cuboid.
        might_mark_freespace_list: List of free spaces to be marked in the image.
        rectangle_grey: Whether to draw the rectangle in grey color.
        save_path: The path to save the rendered image.
        trans_visiblity: The visibility of the transparent items in the scene, 0 is invisible, 1 is fully visible.

    Returns:
        img: The rendered image.
        Note that the image must be convert to RGB format before saving, or the transparency will have bug.

    """

    for entity in scene.entities:
        if entity.name not in transparent_item_list:
            continue
        for component in entity.get_components():
            if isinstance(component, sapien.pysapien.render.RenderBodyComponent):
                component.visibility = trans_visiblity
                scene.step()
                scene.update_render()

    camera = create_and_mount_camera(
        scene,
        pose=pose,
        near=0.1,
        far=1000,
        width=width,
        height=height,
        fovy=fovy,
        camera_name="camera",
    )
    img = render_image(scene, camera)

    circled_numbers = [
        "①",
        "②",
        "③",
        "④",
        "⑤",
        "⑥",
        "⑦",
        "⑧",
        "⑨",
        "⑩",
        "⑪",
        "⑫",
        "⑬",
        "⑭",
        "⑮",
        "⑯",
        "⑰",
        "⑱",
        "⑲",
        "⑳",
        "㉑",
        "㉒",
        "㉓",
        "㉔",
        "㉕",
        "㉖",
        "㉗",
        "㉘",
        "㉙",
        "㉚",
        "㉛",
        "㉜",
        "㉝",
        "㉞",
        "㉟",
    ]

    # 用这个函数替换你当前的颜色获取代码
    colors = get_high_contrast_colors()
    number_idx = 0
    font = ImageFont.truetype(
        "msmincho.ttc", 40 if len(might_mark_object_cuboid_list) <= 10 else 25
    )
    img = add_guide_line_labels(
        img,
        might_mark_object_cuboid_list,
        camera,
        width,
        height,
        colors,
        rectangle_grey,
    )

    for might_mark_freespace in might_mark_freespace_list:
        freespace_image_points = coordinate_convertor.world_rectangle_to_image_polygon(
            might_mark_freespace,
            camera.get_global_pose().p,
            transforms3d.quaternions.quat2mat(camera.get_global_pose().q),
            camera.fovx,
            width,
            height,
        )
        if True:
            # Process the cuboid further freespace

            color = colors[number_idx % len(colors)]
            if freespace_image_points is None:
                continue
            img = draw_freespace_on_image(
                img, freespace_image_points, color, width, height, draw=True
            )

            freespace_image_points = [
                (
                    max(0, min(width - 1, int(point[0]))),
                    max(0, min(height - 1, int(point[1]))),
                )
                for point in freespace_image_points
            ]

            mid_freespace = np.mean(freespace_image_points, axis=0)
            bold_pixel = [(0, 0), (1, 0)]

            font = ImageFont.truetype("msmincho.ttc", int(20))
            draw = ImageDraw.Draw(img)
            for bold_offset in bold_pixel:
                draw.text(
                    tuple(mid_freespace + np.array(bold_offset)),
                    circled_numbers[number_idx % len(circled_numbers)],
                    font=font,
                    fill="black",
                )
            number_idx += 1

    img = img.convert("RGB")

    img.save(save_path)
    camera.disable()

    for entity in scene.entities:
        if entity.name not in transparent_item_list:
            continue
        for component in entity.get_components():
            if isinstance(component, sapien.pysapien.render.RenderBodyComponent):
                component.visibility = 1.0

    # import ipdb
    # ipdb.set_trace()

    return img


def draw_optimized_cuboid_and_guideline(
    img, cuboid_image_points, label_position, obj_id, obj_color=None
):
    """
    优化绘制包围盒和引导线

    Args:
        img: 图像
        cuboid_image_points: 包围盒顶点
        label_position: 标签位置
        obj_id: 物体ID
        obj_color: 物体特定颜色（如果为None则使用默认颜色）
    """
    # 1. 为每个物体选择特定颜色
    if obj_color is None:
        # 使用一个固定的颜色映射，确保相同ID总是获得相同颜色
        colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
            (200, 150, 100),
            (150, 100, 200),
            (100, 200, 150),
        ]
        obj_color = colors[obj_id % len(colors)]

    # 2. 绘制半透明的包围盒
    # 创建多边形顶点列表用于填充
    polygons = []

    # 底面
    bottom_face = np.array(cuboid_image_points[0:4]).astype(np.int32)
    polygons.append(bottom_face)

    # 顶面
    top_face = np.array(cuboid_image_points[4:8]).astype(np.int32)
    polygons.append(top_face)

    # 侧面
    for i in range(4):
        side_face = np.array(
            [
                cuboid_image_points[i],
                cuboid_image_points[(i + 1) % 4],
                cuboid_image_points[(i + 1) % 4 + 4],
                cuboid_image_points[i + 4],
            ]
        ).astype(np.int32)
        polygons.append(side_face)

    # # 半透明填充
    # overlay = img.copy()
    # for poly in polygons:
    #     cv2.fillPoly(overlay, [poly], (obj_color[0], obj_color[1], obj_color[2], 50))

    # # 将半透明层与原图合并
    # alpha = 0.3
    # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # # 3. 使用虚线绘制包围盒边缘
    # edges = [
    #     (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    #     (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
    #     (0, 4), (1, 5), (2, 6), (3, 7)   # 连接线
    # ]

    # for edge in edges:
    #     pt1 = tuple(map(int, cuboid_image_points[edge[0]]))
    #     pt2 = tuple(map(int, cuboid_image_points[edge[1]]))

    #     # 绘制虚线
    #     img = draw_dashed_line(img, pt1, pt2, obj_color, 1, 5, 3)

    # 4. 添加明显的锚点（包围盒顶部中心）
    anchor_point = np.mean(cuboid_image_points[4:8], axis=0).astype(int)
    img_array = np.array(img)
    cv2.circle(img_array, tuple(anchor_point), 4, obj_color, -1)
    img = Image.fromarray(img_array)

    # 5. 绘制从锚点到标签的曲线引导线
    img = draw_curved_guide_line(img, anchor_point, label_position, obj_color)

    return img, anchor_point


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=5):
    """绘制虚线"""
    # 将PIL图像转换为OpenCV格式

    img_array = np.array(img)

    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    if dist <= 0:
        return img

    # 计算方向向量
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    # 绘制虚线
    gap = True
    curr_pos = 0
    while curr_pos < dist:
        curr_pt1 = (
            int(pt1[0] + dx * curr_pos + 0.5),
            int(pt1[1] + dy * curr_pos + 0.5),
        )

        # 确定段长度
        seg_length = gap_length if gap else dash_length
        next_pos = min(curr_pos + seg_length, dist)

        curr_pt2 = (
            int(pt1[0] + dx * next_pos + 0.5),
            int(pt1[1] + dy * next_pos + 0.5),
        )

        # 只绘制实线部分
        if not gap:
            # 确保坐标在图像范围内
            if (
                0 <= curr_pt1[0] < img_array.shape[1]
                and 0 <= curr_pt1[1] < img_array.shape[0]
                and 0 <= curr_pt2[0] < img_array.shape[1]
                and 0 <= curr_pt2[1] < img_array.shape[0]
            ):
                cv2.line(img_array, curr_pt1, curr_pt2, color, thickness)

        # 切换状态
        gap = not gap
        curr_pos = next_pos

    # 转回PIL图像
    img = Image.fromarray(img_array)
    return img


def draw_curved_guide_line(img, start_pt, end_pt, color, thickness=4, num_segments=20):
    """绘制曲线引导线"""
    # 计算控制点
    # 使用贝塞尔曲线，控制点位于起点和终点之间，稍微偏向一侧
    mid_x = (start_pt[0] + end_pt[0]) / 2
    mid_y = (start_pt[1] + end_pt[1]) / 2
    img_array = np.array(img)

    # 添加一些偏移以创建曲线效果
    offset_x = (end_pt[1] - start_pt[1]) * 0.2
    offset_y = (start_pt[0] - end_pt[0]) * 0.2

    control_pt = (int(mid_x + offset_x), int(mid_y + offset_y))

    # 绘制贝塞尔曲线
    prev_pt = start_pt
    for i in range(1, num_segments + 1):
        t = i / num_segments
        # 二次贝塞尔曲线
        x = int(
            (1 - t) ** 2 * start_pt[0]
            + 2 * (1 - t) * t * control_pt[0]
            + t**2 * end_pt[0]
            + 0.5
        )
        y = int(
            (1 - t) ** 2 * start_pt[1]
            + 2 * (1 - t) * t * control_pt[1]
            + t**2 * end_pt[1]
            + 0.5
        )

        curr_pt = (x, y)

        # 线条从粗到细，让起点更明显
        curr_thickness = max(thickness, int(thickness * (1 - t * 0.7)))
        cv2.line(img_array, prev_pt, curr_pt, color, curr_thickness)

        prev_pt = curr_pt

    img = Image.fromarray(img_array)
    return img


def add_guide_line_labels(
    img,
    might_mark_object_cuboid_list,
    camera,
    width,
    height,
    colors,
    rectangle_grey=False,
):
    def optimize_sequence(seq, min_diff, min_val, max_val):
        n = len(seq)
        new_seq = np.array(seq)
        for _ in range(5):
            for i in range(n - 1):
                if new_seq[i + 1] - new_seq[i] < min_diff:
                    new_seq[i + 1] = new_seq[i] + min_diff
            shift = max(0, new_seq[-1] - max_val)
            if shift > 0:
                new_seq -= shift
            for i in range(n - 1, 0, -1):
                if new_seq[i] - new_seq[i - 1] < min_diff:
                    new_seq[i - 1] = new_seq[i] - min_diff
            shift = max(0, min_val - new_seq[0])
            if shift > 0:
                new_seq += shift
        import ipdb

        ipdb.set_trace()

        return new_seq.tolist()

    """主函数：使用优化后的方法绘制包围盒和引导线"""
    # 存储已使用的标签位置
    used_positions = []

    # 存储每个物体的引导线起点，用于后续优化
    anchor_points = []
    label_info = []

    # 步骤1: 先绘制所有包围盒(不含引导线)
    number_idx = 0
    need_optimize_label_idx_list = []
    need_optimize_label_list = []
    for cuboid_idx, might_mark_cuboid in enumerate(might_mark_object_cuboid_list):
        if len(might_mark_cuboid) == 0:
            continue

        # 计算包围盒各顶点在图像中的位置
        cuboid_image_points = [
            coordinate_convertor.world_to_image(
                point,
                camera.get_global_pose().p,
                transforms3d.quaternions.quat2mat(camera.get_global_pose().q),
                camera.fovx,
                width,
                height,
            )
            for point in might_mark_cuboid
        ]

        # 选择颜色
        base_color = colors[cuboid_idx % len(colors)]

        # 计算包围盒的中心点(用于确定标签位置)
        center_2d = np.mean(cuboid_image_points, axis=0)

        # 选择最佳区域放置标签
        region = select_best_region(center_2d, width, height)

        # 找到适合的标签位置
        label_position, _ = find_label_position(
            region, used_positions, center_2d, 50, width, height
        )

        if _:
            need_optimize_label_idx_list.append(number_idx)
        used_positions.append(label_position)

        # 保存信息以便步骤2使用
        label_info.append(
            {
                "id": number_idx + 1,
                "cuboid_image_points": cuboid_image_points,
                "label_position": label_position,
                "color": base_color,
            }
        )

        number_idx += 1

    need_optimize_label_list = [
        label_info[i]["label_position"] for i in need_optimize_label_idx_list
    ]
    leveled_list = []
    leveled_id_list = []
    for h, label in enumerate(need_optimize_label_list):
        for i in range(len(leveled_list)):
            if np.abs(leveled_list[i][0][1] - label[1]) < 5:
                leveled_list[i].append(label)
                leveled_id_list[i].append(need_optimize_label_idx_list[h])

                break
            elif i == len(leveled_list) - 1:
                leveled_list.append([label])
    for i, level_list in enumerate(leveled_list):
        seq = [label[0] for label in level_list]
        optimized_seq = optimize_sequence(seq, 50, 0, width)
        for j, label in enumerate(level_list):
            id = leveled_id_list[i][j]
            label_info[id]["label_position"] = (optimized_seq[j], label[1])

    # 步骤2: 优化引导线路径并绘制所有元素
    # 先排序，让距离近的物体先画（减少交叉）
    label_info.sort(key=lambda x: calculate_depth(x["cuboid_image_points"]))

    font = ImageFont.truetype("msmincho.ttc", int(30))
    draw = ImageDraw.Draw(img)

    for info in label_info:
        obj_id = info["id"]
        cuboid_points = info["cuboid_image_points"]
        label_position = info["label_position"]
        color = info["color"]
        import ipdb

        # 绘制优化的包围盒和引导线
        img, anchor_pt = draw_optimized_cuboid_and_guideline(
            img, cuboid_points, label_position, obj_id, color
        )
        # ipdb.set_trace()
        # 绘制标签背景（黑色圆形）
        draw_circle = ImageDraw.Draw(img)
        draw_circle.ellipse(
            [
                (label_position[0] - 20, label_position[1] - 20),
                (label_position[0] + 20, label_position[1] + 20),
            ],
            fill=color,
        )
        # ipdb.set_trace()
        # 绘制编号文本
        draw = ImageDraw.Draw(img)
        draw.text(
            (int(label_position[0] - 15), int(label_position[1] - 15)),
            str(obj_id),
            font=font,
            fill="white",
        )
    # ipdb.set_trace()

    return img


def calculate_depth(cuboid_points):
    """简单计算包围盒的深度（用于排序）"""
    # 使用Z坐标的平均值作为深度估计
    return np.mean([p[2] for p in cuboid_points]) if len(cuboid_points[0]) > 2 else 0


def select_best_region(center_2d, image_width, image_height):
    """选择最佳的标签放置区域"""
    x, y = center_2d

    # 计算点到四边的距离
    dist_to_top = y
    dist_to_bottom = image_height - y
    dist_to_left = x
    dist_to_right = image_width - x

    # 找出最小距离对应的边
    min_dist = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)

    if min_dist == dist_to_top:
        return "top"
    elif min_dist == dist_to_bottom:
        return "bottom"
    elif min_dist == dist_to_left:
        return "left"
    else:
        return "right"


def find_label_position(
    region, used_positions, center_2d, margin, image_width, image_height
):
    """在指定区域中找到不与其他标签重叠的位置，优先考虑上下方位置"""
    x, y = center_2d

    # 如果原始区域是左右两侧，尝试重新分配到上下方
    if region == "left" or region == "right":
        # 判断距离上方还是下方更近
        if y < image_height / 2:
            # 更靠近上方
            new_region = "top"
            initial_pos = (x, margin)
        else:
            # 更靠近下方
            new_region = "bottom"
            initial_pos = (x, image_height - margin)
    else:
        # 保持原有的上下方位置
        if region == "top":
            initial_pos = (x, margin)
        else:  # bottom
            initial_pos = (x, image_height - margin)

    # 检查是否与已有标签重叠
    position = initial_pos
    step = 20  # 标签移动步长
    max_tries = 30  # 最大尝试次数

    # 首先尝试在上下方找到位置
    for _ in range(max_tries):
        if not is_position_overlapping(position, used_positions):
            return position, 1

        # 水平移动 (适用于上下方区域)
        move_direction = 1 if np.random.random() > 0.5 else -1
        position = (position[0] + step * move_direction, position[1])

        # 确保不超出图像边界
        if position[0] < margin:
            position = (margin, position[1])
        elif position[0] > image_width - margin:
            position = (image_width - margin, position[1])

    # 如果在上下方无法找到位置，则尝试更大范围的上下方位置
    # 创建一个在上下方移动的梯度
    vertical_steps = [(0, step), (0, -step)]  # 向下移动  # 向上移动

    position = initial_pos
    for _ in range(max_tries):
        # 随机选择向上或向下移动
        v_step = vertical_steps[np.random.randint(0, 2)]
        position = (position[0], position[1] + v_step[1])

        # 保持在上下边界有一定距离内
        if position[1] < margin:
            position = (position[0], margin)
        elif position[1] > image_height - margin:
            position = (position[0], image_height - margin)

        if not is_position_overlapping(position, used_positions):
            return position, 1

    # 如果上述方法都失败，最后才考虑左右两侧
    if region == "left":
        side_pos = (margin, y)
    elif region == "right":
        side_pos = (image_width - margin, y)
    else:
        # 如果实在找不到位置，返回初始位置
        return initial_pos, 0

    # 在左右两侧尝试避开其他标签
    position = side_pos
    for _ in range(max_tries):
        if not is_position_overlapping(position, used_positions):
            return position, 0

        # 垂直移动
        move_direction = 1 if np.random.random() > 0.5 else -1
        position = (position[0], position[1] + step * move_direction)

        # 确保不超出图像边界
        if position[1] < margin:
            position = (position[0], margin)
        elif position[1] > image_height - margin:
            position = (position[0], image_height - margin)

    # 如果所有尝试都失败，返回初始位置
    return initial_pos, 0


def is_position_overlapping(position, used_positions, threshold=40):
    """检查位置是否与已使用的位置重叠"""

    for used_pos in used_positions:
        distance = np.sqrt(
            (position[0] - used_pos[0]) ** 2 + (position[1] - used_pos[1]) ** 2
        )
        if distance < threshold:
            return True
    return False


def draw_line_on_image(img, pt1, pt2, color, thickness=1):
    """在图像上绘制线段"""
    # 根据你使用的图像库选择合适的方法
    # 如果使用PIL
    draw = ImageDraw.Draw(img)
    draw.line([pt1, pt2], fill=color, width=thickness)

    # 如果使用OpenCV
    # img_array = np.array(img)
    #  cv2.line(img_array, pt1, pt2, color, thickness)

    return img


def draw_circle_on_image(img, center, radius, color, thickness=-1):
    """在图像上绘制圆形"""
    # 根据你使用的图像库选择合适的方法
    # 如果使用PIL
    draw = ImageDraw.Draw(img)
    top_left = (center[0] - radius, center[1] - radius)
    bottom_right = (center[0] + radius, center[1] + radius)
    draw.ellipse(
        [top_left, bottom_right],
        fill=color if thickness == -1 else None,
        outline=color if thickness != -1 else None,
    )

    # 如果使用OpenCV
    # img_array = np.array(img)
    # cv2.circle(img, center, radius, color, thickness)

    return img


def main():
    pass


if __name__ == "__main__":
    main()
