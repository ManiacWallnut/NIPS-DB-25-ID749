import sapien as sapien
from sapien.utils import Viewer
import numpy as np
import json
import transforms3d
import os
from pathlib import Path
import os.path as osp
import argparse
import xml.etree.ElementTree as ET
import open3d as o3d
from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch import (
    FETCH_BASE_COLLISION_BIT,
    FETCH_WHEELS_COLLISION_BIT,
)
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs import Actor, Articulation
from shapely.geometry import Polygon, Point
from scipy.spatial.transform import Rotation as R

# import imagepoint
from . import imagepoint
from PIL import Image, ImageColor, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import copy
import cv2


def urdf_to_dict(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    def parse_element(element):
        parsed = {}
        # if 'name' in element.attrib:
        #    parsed['name'] = element.attrib['name']
        for child in element:
            if len(child) > 0:
                name = f"_{child.attrib['name']}" if "name" in child.attrib else ""
                parsed[f"{child.tag}{name}"] = parse_element(child)
            else:
                parsed[child.tag] = child.attrib
        return parsed

    urdf_dict = {root.tag: parse_element(root)}
    return urdf_dict


def load_objects_from_json(scene, json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # load background
    q = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), theta=np.deg2rad(90))
    bg_pose = sapien.Pose(q=q)
    bg_path = data["background_file_path"]

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(bg_path)
    builder.add_nonconvex_collision_from_file(bg_path)
    bg = builder.build_static(name=f"scene_background")
    bg.set_pose(bg_pose)

    # load objects
    id_map = dict()

    for obj in data["object_instances"]:

        if obj["name"] == "GROUND":
            continue

        object_file_path = obj["visual_path"]
        collision_path = obj["collision_path"]
        position = [
            obj["centroid_translation"]["x"],
            obj["centroid_translation"]["y"],
            obj["centroid_translation"]["z"],
        ]

        quaternion = [
            obj["quaternion"]["w"],
            obj["quaternion"]["x"],
            obj["quaternion"]["y"],
            obj["quaternion"]["z"],
        ]

        rpy = transforms3d.euler.quat2euler(quaternion, axes="sxyz")
        quaternion = transforms3d.euler.euler2quat(
            rpy[0] + np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
        )

        builder = scene.create_actor_builder()
        builder.add_visual_from_file(filename=object_file_path)
        if collision_path is not None:
            builder.add_multiple_convex_collisions_from_file(filename=collision_path)
        else:

            builder.add_convex_collision_from_file(filename=object_file_path)

        mesh = builder.build(name=obj["name"])
        mesh.set_pose(sapien.Pose(p=position, q=quaternion))

    for articulated_meta in data.get("articulate_instances", []):
        template_name = articulated_meta["name"]
        env_idx = int(template_name[template_name.rfind("_") + 1 :])
        if "door" in template_name:
            continue
        pos = articulated_meta["translation"]
        pos = [pos["x"], pos["y"], pos["z"]]

        rot = articulated_meta["rotation"]
        rot = [rot["w"], rot["x"], rot["y"], rot["z"]]

        rpy = transforms3d.euler.quat2euler(rot, axes="sxyz")
        quaternion = transforms3d.euler.euler2quat(
            rpy[0] + np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
        )

        urdf_path = articulated_meta["urdf_path"]
        urdf_loader = scene.create_urdf_loader()
        articulation_name = template_name
        urdf_loader.name = f"{articulation_name}"
        urdf_loader.fix_root_link = articulated_meta["fixed_base"]
        urdf_loader.disable_self_collisions = True
        if "uniform_scale" in articulated_meta:
            urdf_loader.scale = articulated_meta["uniform_scale"]
        builder = urdf_loader.parse(urdf_path)[0][0]
        pose = sapien.Pose(pos, quaternion)
        builder.initial_pose = pose
        articulation = builder.build()


def load_articulations(scene):
    q = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), theta=np.deg2rad(90))

    articulations: dict[str, Articulation] = dict()

    bg_pose = sapien.Pose(q=q)


REPLICA_DATASET_ROOT_PATH = Path("./scene_datasets/replica_cad_dataset")


def get_glb_path(template_name):
    # object config
    obj_config_path = (
        REPLICA_DATASET_ROOT_PATH
        / "configs/objects"
        / f"{osp.basename(template_name)}.object_config.json"
    )
    # print(obj_config_path)
    with open(obj_config_path, "r") as f:
        obj_config = json.load(f)

    # object glb file path from config
    relative_glb_path = obj_config["render_asset"]
    glb_file_path = os.path.normpath(obj_config_path.parent / relative_glb_path)

    return glb_file_path


def reget_entities_from_sapien(
    scene,
    json_file_path,
    path="entities.json",
    visual_path_prefix="scene_datasets/replica_cad_dataset/objects/",
    urdf_path_prefix="scene_datasets/replica_cad_dataset/urdf/",
):
    def convert_to_float(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_float(i) for i in obj]
        return obj

    res = {}
    res["background_file_path"] = (
        "scene_datasets/replica_cad_dataset/stages/frl_apartment_stage.glb"
    )

    data = []

    with open(json_file_path, "r") as f:
        data = json.load(f)

    articulations = data.get("articulate_instances", [])
    articulation_idx = -1

    res["object_instances"] = []
    for entity in scene.entities:
        entity_instance = {}

        if entity.get_name() == "root":
            articulation_idx += 1
            continue

        if articulation_idx == -1:
            entity_instance["name"] = entity.get_name()
            filename = entity.get_name()
            filename = filename[: filename.rfind("_")]

            entity_instance["visual_path"] = f"{visual_path_prefix}{filename}.glb"
        elif articulation_idx < len(articulations):
            entity_instance["name"] = (
                f'{articulations[articulation_idx]["name"]}_{entity.get_name()}'
            )
            urdf_path = articulations[articulation_idx]["urdf_path"]
            urdf_dict = urdf_to_dict(urdf_path)
            if f"link_{entity.get_name()}" not in urdf_dict["robot"]:
                print(f"Warning: {entity.get_name()} not found in URDF")

                continue
            filename = urdf_dict["robot"][f"link_{entity.get_name()}"]["visual"][
                "geometry"
            ]["mesh"]["filename"]
            name = articulations[articulation_idx]["name"]

            name = name[: name.rfind("_")]
            entity_instance["visual_path"] = f"{urdf_path_prefix}{name}/{filename}"
        else:
            print("Error: articulation index out of range")
        quaternion = [convert_to_float(val) for val in entity.get_pose().q]
        rpy = transforms3d.euler.quat2euler(quaternion, axes="sxyz")
        quaternion = transforms3d.euler.euler2quat(
            rpy[0] - np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
        )
        q = {
            "w": convert_to_float(quaternion[0]),
            "x": convert_to_float(quaternion[1]),
            "y": convert_to_float(quaternion[2]),
            "z": convert_to_float(quaternion[3]),
        }

        p = {
            "x": convert_to_float(entity.get_pose().p[0]),
            "y": convert_to_float(entity.get_pose().p[1]),
            "z": convert_to_float(entity.get_pose().p[2]),
        }

        # q = {'w': convert_to_float(entity.get_pose().q[0]), 'x': convert_to_float(entity.get_pose().q[1]), 'y': convert_to_float(entity.get_pose().q[2]), 'z': convert_to_float(entity.get_pose().q[3])}
        entity_instance["centroid_translation"] = p
        entity_instance["quaternion"] = q
        entity_instance["bbox"] = "deprecated"
        res["object_instances"].append(entity_instance)

    with open(path, "w") as f:
        json.dump(res, f, indent=4)


def create_camera(
    scene,
    pose: sapien.Pose,
    near: float,
    far: float,
    width: float = 1920,
    height: float = 1080,
    fovy: float = np.deg2rad(60),
    camera_name: str = "camera",
):
    camera = scene.add_camera(
        name=camera_name,
        pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        width=width,
        height=height,
        fovy=fovy,
        near=near,
        far=far,
    )
    camera.set_pose(pose)

    return camera


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
    pass


def render_image(scene, camera):
    scene.step()
    scene.step()
    scene.update_render()
    camera.take_picture()
    #'get_picture' function requires the color type. 'normal', 'color', 'depth', 'segmentation' etc.
    rgba = camera.get_picture("Color")
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(rgba_img)

    return img


def box_on_image(height, width, box):

    def inside_box(x, y, box):
        n = len(box)

        cross_sign = [
            (box[i][0] - x) * (box[(i + 1) % n][1] - y)
            - (box[i][1] - y) * (box[(i + 1) % n][0] - x)
            for i in range(n)
        ]
        return all(
            [cross_sign[i] * cross_sign[i + 1] > 0 for i in range(n - 1)]
        ) or all([cross_sign[i] * cross_sign[i + 1] < 0 for i in range(n - 1)])

    #    print('box',box)
    points = []
    l = int(min(point[0] for point in box))
    r = int(max(point[0] for point in box))
    u = int(min(point[1] for point in box))
    d = int(max(point[1] for point in box))
    if len(box) < 4 or l < -1e5 or r > 1e5 or u < -1e5 or d > 1e5:
        return []

    box_area = np.cross(
        np.array(box[1]) - np.array(box[0]), np.array(box[2]) - np.array(box[0])
    )
    if abs(box_area) < 1e-5:
        return []
    l = max(0, l)
    r = min(width - 1, r)
    u = max(0, u)
    d = min(height - 1, d)
    for i in range(l, r + 1):
        for j in range(u, d + 1):
            if inside_box(i, j, box):
                points.append((i, j))
    return points


def free_space_boxes_on_image(height, width, free_spaces):

    def inside_box(x, y, box):
        n = len(box)

        cross_sign = [
            (box[i][0] - x) * (box[(i + 1) % n][1] - y)
            - (box[i][1] - y) * (box[(i + 1) % n][0] - x)
            for i in range(n)
        ]
        return all(
            [cross_sign[i] * cross_sign[i + 1] > 0 for i in range(n - 1)]
        ) or all([cross_sign[i] * cross_sign[i + 1] < 0 for i in range(n - 1)])

    def cal_box_content(box):
        points = []
        l = int(min(point[0] for point in box))
        r = int(max(point[0] for point in box))
        u = int(min(point[1] for point in box))
        d = int(max(point[1] for point in box))
        if l < -1e5 or r > 1e5 or u < -1e5 or d > 1e5:
            return []

        box_area = np.cross(
            np.array(box[1]) - np.array(box[0]), np.array(box[2]) - np.array(box[0])
        )
        if abs(box_area) < 1e-5:
            return []
        l = max(0, l)
        r = min(width - 1, r)
        u = max(0, u)
        d = min(height - 1, d)
        for i in range(l, r + 1):
            for j in range(u, d + 1):
                if inside_box(i, j, box):
                    points.append((i, j))
        return points

    front_points, back_points, left_points, right_points = [], [], [], []
    rear_left_points, rear_right_points, front_left_points, front_right_points = (
        [],
        [],
        [],
        [],
    )

    if "front" in free_spaces:
        front_points = cal_box_content(free_spaces["front"])
    if "rear" in free_spaces:
        back_points = cal_box_content(free_spaces["rear"])
    if "left" in free_spaces:
        left_points = cal_box_content(free_spaces["left"])
    if "right" in free_spaces:
        right_points = cal_box_content(free_spaces["right"])
    if "rear-left" in free_spaces:
        rear_left_points = cal_box_content(free_spaces["rear-left"])
    if "rear-right" in free_spaces:
        rear_right_points = cal_box_content(free_spaces["rear-right"])
    if "front-left" in free_spaces:
        front_left_points = cal_box_content(free_spaces["front-left"])
    if "front-right" in free_spaces:
        front_right_points = cal_box_content(free_spaces["front-right"])

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

    pass


def vector_to_rpy(vector):

    # 归一化向量
    vector = vector / np.linalg.norm(vector)

    roll_value = np.arctan2(vector[1], vector[0])
    pitch_value = np.arcsin(vector[2])

    rpy = [roll_value, pitch_value, 0]

    return rpy


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
            imagepoint.world_to_image(
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
        mid = imagepoint.world_to_image(
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
        rpy = vector_to_rpy(top_center - free_space_center)
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
                    imagepoint.world_to_image(
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
                    imagepoint.world_to_image(
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
                    imagepoint.world_to_image(
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
                    imagepoint.world_to_image(
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
                imagepoint.world_to_image(
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
                imagepoint.world_to_image(
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
                imagepoint.world_to_image(
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
                imagepoint.world_to_image(
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
                mid = imagepoint.world_to_image(
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
                mid = imagepoint.world_to_image(
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
                mid = imagepoint.world_to_image(
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
                mid = imagepoint.world_to_image(
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


def calculate_area_from_image(
    rect_points, camera_pose, camera_rotation, fovx, width, height
):

    # 将矩形的四个顶点转换到图像坐标系
    # def world_to_image(world_point, camera_position, camera_rotation, fovx, image_width, image_height):
    image_points = [
        imagepoint.world_to_image(
            point, camera_pose, camera_rotation, fovx, width, height
        )
        for point in rect_points
    ]
    for point in image_points:
        if abs(point[0]) > 1e5 or abs(point[1]) > 1e5:
            return 0

    xs = [point[0] for point in image_points]
    ys = [point[1] for point in image_points]
    xs = np.sort(xs)
    ys = np.sort(ys)

    sign = 0
    if xs[0] > 0 and xs[3] < width and ys[0] > 0 and ys[3] < height:
        sign = 1

    # ensure
    n_piece, m_piece = 3, 8

    area = 0

    img_rect_array = [
        [
            np.array(
                [
                    [i * width // n_piece, j * height // m_piece],
                    [(i + 1) * width // n_piece - 1, j * height // m_piece],
                    [(i + 1) * width // n_piece - 1, (j + 1) * height // m_piece - 1],
                    [i * width // n_piece, (j + 1) * height // m_piece - 1],
                ]
            )
            for j in range(m_piece)
        ]
        for i in range(n_piece)
    ]

    for i in range(n_piece):
        for j in range(m_piece):
            intersection_ij = cv2.intersectConvexConvex(
                np.array(image_points), img_rect_array[i][j]
            )
            if intersection_ij[1] is not None:
                area_ij = cv2.contourArea(intersection_ij[1])
                area += area_ij / (
                    (i - (n_piece + 1) * 0.5) ** 2 + (j - (m_piece + 1) * 0.5) ** 2
                )

    # 计算交集矩形的面积

    return area * sign


def calculate_rect_area_in_image(
    camera_pose, camera_angles, fovy, rect_points, width, height
):
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
    rpy = [0, pitch, yaw]
    quat = R.from_euler("xyz", rpy).as_quat()
    camera_rotation = R.from_quat(quat).as_matrix()

    fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)

    # 计算矩形在图像中占据的面积
    area = calculate_area_from_image(
        rect_points, camera_pose, camera_rotation, fovx, width, height
    )

    return area


def annealing_search(f, left, right, initial=None, temperature=1, cooling_rate=0.9):

    current = initial if initial is not None else (left + right) / 2
    best = current
    best_value = f(current)

    while temperature > 1e-3:
        new_current = current + np.random.normal(0, temperature)
        if new_current < left or new_current > right:
            temperature *= cooling_rate**1 / 2
            continue
        new_value = f(new_current)
        if new_value > best_value:
            best = new_current
            best_value = new_value
        if new_value > f(current) or (
            new_value != 0
            and np.random.rand() < np.exp((new_value - f(current)) / temperature)
        ):
            current = new_current
        temperature *= cooling_rate

    return best


def find_initial_angle_range(camera_pose, rect_center):

    direction = rect_center - camera_pose
    direction = direction / np.linalg.norm(direction)

    pitch = -np.arcsin(direction[2])
    yaw = np.arctan2(direction[1], direction[0])

    pitch_range = [pitch - np.pi / 2, pitch + np.pi / 2]
    yaw_range = [yaw - np.pi, yaw + np.pi]
    return pitch_range, yaw_range


def find_optimal_camera_pose(
    scene,
    rect_points,
    fixed_x,
    fixed_y,
    z_min,
    z_max,
    fovy_min,
    fovy_max,
    width,
    height,
    absolute_precision=1e-3,
    rad_precision=0.5,
    must_top_view=False,
):

    rect_center = np.mean(rect_points, axis=0)
    camera_pose = np.array([fixed_x, fixed_y, z_max])

    best_pitch, best_yaw = None, None
    best_fovy = fovy_min
    best_z = z_max

    while best_pitch is None or best_yaw is None:
        best_pitch, best_yaw = imagepoint.find_valid_pitch_yaw(
            required_points=rect_points,
            camera_pose=camera_pose,
            fovy=best_fovy,
            width=width,
            height=height,
            must_top_view=must_top_view,
        )
        if best_pitch is None or best_yaw is None:
            best_fovy = best_fovy + (
                np.deg2rad(10) if best_fovy < fovy_max else np.deg2rad(3)
            )
            if best_fovy > np.deg2rad(150):
                camera_pose[2] += 0.5
            if best_fovy > np.deg2rad(179):
                print("failed to find fovy")
                return best_yaw, best_pitch, best_z, np.deg2rad(150)

    # print('initial area', calculate_rect_area_in_image(np.array([fixed_x, fixed_y, best_z]), (best_pitch, best_yaw), best_fovy, rect_points, width, height))

    yaw_range = [best_yaw - np.pi / 2, best_yaw + np.pi / 2]
    pitch_range = [-np.pi / 2, np.pi / 10]
    if must_top_view:
        pitch_range = [-np.pi / 2 - rad_precision, -np.pi / 2 + rad_precision]

    best_yaw = annealing_search(
        lambda yaw: calculate_rect_area_in_image(
            np.array([fixed_x, fixed_y, best_z]),
            (best_pitch, yaw),
            best_fovy,
            rect_points,
            width,
            height,
        ),
        yaw_range[0],
        yaw_range[1],
        best_yaw,
        np.pi / 2,
        0.6,
    )

    #  print('best yaw area', calculate_rect_area_in_image(np.array([fixed_x, fixed_y, best_z]), (best_pitch, best_yaw), best_fovy, rect_points, width, height))

    best_pitch = annealing_search(
        lambda pitch: calculate_rect_area_in_image(
            np.array([fixed_x, fixed_y, best_z]),
            (pitch, best_yaw),
            best_fovy,
            rect_points,
            width,
            height,
        ),
        pitch_range[0],
        pitch_range[1],
        best_pitch,
        np.pi / 2,
        0.6,
    )

    #  print('best pitch area', calculate_rect_area_in_image(np.array([fixed_x, fixed_y, best_z]), (best_pitch, best_yaw), best_fovy, rect_points, width, height))

    best_fovy = annealing_search(
        lambda fovy: calculate_rect_area_in_image(
            np.array([fixed_x, fixed_y, best_z]),
            (best_pitch, best_yaw),
            fovy,
            rect_points,
            width,
            height,
        ),
        fovy_min,
        fovy_max,
        best_fovy,
        np.pi / 3,
        0.6,
    )
    #  print('best fovy area', calculate_rect_area_in_image(np.array([fixed_x, fixed_y, best_z]), (best_pitch, best_yaw), best_fovy, rect_points, width, height))

    best_z = annealing_search(
        lambda z: calculate_rect_area_in_image(
            np.array([fixed_x, fixed_y, z]),
            (best_pitch, best_yaw),
            best_fovy,
            rect_points,
            width,
            height,
        ),
        z_min,
        z_max,
        best_z,
        1,
        0.6,
    )
    #  print('best z area', calculate_rect_area_in_image(np.array([fixed_x, fixed_y, best_z]), (best_pitch, best_yaw), best_fovy, rect_points, width, height))

    return best_yaw, best_pitch, best_z, best_fovy

    pass


def separate_points(points, min_distance, img_width, img_height):
    """
    Separate points that are too close to each other.

    :param points: List of (x, y) tuples representing the points.
    :param min_distance: Minimum allowed distance between points.
    :param img_width: Width of the image.
    :param img_height: Height of the image.
    :return: List of separated points.
    """
    points = np.array(points)
    num_points = len(points)
    if min_distance < 1e-6 or num_points < 2:
        return points.tolist()

    grid_size = min_distance

    grid = {}

    def get_grid_position(point):
        return (int(point[0] // grid_size), int(point[1] // grid_size))

    def is_valid_position(point):
        x, y = point
        return 0 <= x < img_width and 0 <= y < img_height

    for i in range(num_points):
        grid_pos = get_grid_position(points[i])
        if grid_pos not in grid:
            grid[grid_pos] = points[i]
        else:
            # Find a new position for the point
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    new_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
                    new_point = points[i] + np.array([dx * grid_size, dy * grid_size])
                    if new_pos not in grid and is_valid_position(new_point):
                        grid[new_pos] = new_point
                        points[i] = new_point
                        break

    return list(grid.values())


def auto_render_image(
    scene,
    x,
    y,
    z_min,
    z_max,
    fovy_min,
    fovy_max,
    necessary_bbox,
    mark_points,
    mark_rectangles,
    width=1920,
    height=1080,
    rectangle_grey=False,
    save_path: str = "image.png",
    must_top_view=False,
):
    # auto render image
    # note: mark_points and mark_rectangles are 3d points!!

    best_yaw, best_pitch, best_z, best_fovy = find_optimal_camera_pose(
        scene,
        rect_points=necessary_bbox,
        fixed_x=x,
        fixed_y=y,
        z_min=z_min,
        z_max=z_max,
        fovy_min=fovy_min,
        fovy_max=fovy_max,
        width=width,
        height=height,
        absolute_precision=1e-2,
        rad_precision=1e-4,
        must_top_view=must_top_view,
    )

    if best_yaw is None or best_pitch is None or best_z is None or best_fovy is None:
        print("Warning: unable to contain all the necessary points.")
        best_yaw, best_pitch, best_z, best_fovy = find_optimal_camera_pose(
            scene,
            rect_points=[necessary_bbox[0] for i in range(4)],
            fixed_x=x,
            fixed_y=y,
            z_min=z_min,
            z_max=z_max,
            fovy_min=fovy_min,
            fovy_max=fovy_max,
            width=width,
            height=height,
            absolute_precision=1e-2,
            rad_precision=1e-4,
            must_top_view=must_top_view,
        )

    if must_top_view:
        pitch_range, yaw_range = find_initial_angle_range(
            camera_pose=np.array([x, y, best_z]), rect_center=necessary_bbox[0]
        )

        best_pitch = np.deg2rad(90)
        best_yaw = -(yaw_range[0] + (yaw_range[1] - yaw_range[0]) / 2)
        best_fovy = np.deg2rad(90)
        best_z = z_max

    if best_fovy > np.deg2rad(150):
        print(
            "Warning: a picture have to use extremely large fovy to satisfy the requirements"
        )

    quat = transforms3d.euler.euler2quat(0, best_pitch, best_yaw, axes="sxyz")
    camera_pose = np.array([x, y, best_z])
    camera = create_and_mount_camera(
        scene,
        pose=sapien.Pose(p=camera_pose, q=quat),
        near=0.1,
        far=1000,
        width=width,
        height=height,
        fovy=best_fovy,
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

    colors = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
    colors = [tuple([int(color[i] * 255) for i in range(3)]) for color in colors]

    font = ImageFont.truetype("msmincho.ttc", 40 if len(mark_points) <= 10 else 25)

    dummy = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)
    text_sizes = [
        draw.textbbox((0, 0), circled_numbers[i], font=font)
        for i in range(len(mark_points))
    ]
    text_size = (
        max(size[2] - size[0] for size in text_sizes) if len(text_sizes) > 0 else 0
    )

    min_distance_for_mark_points = text_size

    #    min_distance_for_mark_points = np.sqrt(text_size[0] ** 2 + text_size[1] ** 2) / 2

    n_mark_points = len(mark_points)
    n_mark_rectangles = len(mark_rectangles)
    # assert n_mark_points <= 35 and len(mark_rectangles) == n_mark_points

    mark_numbers = []

    for i in range(n_mark_rectangles):

        if len(mark_rectangles[i]) < 4:
            continue

        points = [
            imagepoint.world_to_image(
                point,
                camera.get_global_pose().p,
                transforms3d.quaternions.quat2mat(camera.get_global_pose().q),
                camera.fovx,
                width,
                height,
            )
            for point in mark_rectangles[i]
        ]
        pixels = box_on_image(height, width, points)

        if len(pixels) > 0:
            mark_numbers.append(i)

        for pixel in pixels:
            current_color = img.getpixel(pixel)
            color = (
                colors[i % len(colors)] if rectangle_grey == False else (160, 160, 160)
            )

            new_color = tuple([current_color[j] // 2 + color[j] // 2 for j in range(3)])

            if not rectangle_grey or (pixel[0] + pixel[1]) % 2 == 0:
                img.putpixel(pixel, new_color)

    mark_points_2d = [
        imagepoint.world_to_image(
            mark_points[i],
            camera.get_global_pose().p,
            transforms3d.quaternions.quat2mat(camera.get_global_pose().q),
            camera.fovx,
            width,
            height,
        )
        for i in range(n_mark_points)
    ]

    min_distance_in_points = width
    for i in range(n_mark_points):
        for j in range(i + 1, n_mark_points):
            distance = np.linalg.norm(
                np.array(mark_points_2d[i]) - np.array(mark_points_2d[j])
            )
            if distance < min_distance_in_points:
                min_distance_in_points = distance
    min_distance_in_points = int(np.clip(min_distance_in_points, 20, 40))
    bold_pixel = [(-1, 0), (0, -1), (-1, -1), (0, 0), (1, 0), (0, 1), (1, 1)]
    if min_distance_in_points < 25:
        bold_pixel = [(0, 0), (1, 0)]

    font = ImageFont.truetype("msmincho.ttc", int(min_distance_in_points))
    # mark_points_2d = separate_points(mark_points_2d, min_distance_for_mark_points, width, height)

    for i in range(n_mark_points):
        point = mark_points_2d[i]

        draw = ImageDraw.Draw(img)

        for bold_offset in bold_pixel:
            draw.text(
                tuple(np.array(point) + np.array(bold_offset)),
                circled_numbers[i],
                font=font,
                fill="black",
            )

    print(
        "Render fin. save_path",
        save_path,
        "area",
        calculate_rect_area_in_image(
            camera_pose,
            (best_pitch, best_yaw),
            best_fovy,
            necessary_bbox,
            width,
            height,
        ),
    )
    img.save(save_path)
    camera.disable()

    return img, mark_numbers


def get_human_like_view_angles(table_center, camera_pos):
    """
    获取类人视角的相机角度范围

    Args:
        table_center: 桌子中心点坐标 [x, y, z]
        camera_pos: 相机位置 [x, y, z]

    Returns:
        roll_range: roll角度范围 [min, max]
        pitch_range: pitch角度范围 [min, max]
        yaw_range: yaw角度范围 [min, max]
    """
    # 1. Roll角度限制
    roll_range = [-np.pi / 12, np.pi / 12]  # 允许轻微倾斜,±15度

    # 2. Pitch角度限制(俯仰角)
    pitch_range = [-np.pi / 3, np.pi / 6]  # -60度到30度
    # -60度保证不会仰视
    # +30度保证不会太俯视

    # 3. Yaw角度(水平旋转)
    # 计算相机指向桌子的大致方向
    direction = table_center - camera_pos
    base_yaw = np.arctan2(direction[1], direction[0])

    # 以基准方向为中心,左右90度范围
    yaw_range = [base_yaw - np.pi / 2, base_yaw + np.pi / 2]

    return roll_range, pitch_range, yaw_range


def solve_camera_pose_for_4_points(
    points,
    camera_xyz,
    width,
    height,
    focus_ratio,
    fovy_range=[np.deg2rad(5), np.deg2rad(100)],
):

    point_center = np.mean(points, axis=0)

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
            np.array([-width / 2, height / 2]) * focus_ratio
            + np.array([width / 2, height / 2]),
            np.array([width / 2, height / 2]) * focus_ratio
            + np.array([width / 2, height / 2]),
            np.array([-width / 2, -height / 2]) * focus_ratio
            + np.array([width / 2, height / 2]),
            np.array([+width / 2, -height / 2]) * focus_ratio
            + np.array([width / 2, height / 2]),
        ]
        fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)
        total_error = np.sum(
            [
                np.linalg.norm(
                    imagepoint.world_to_image(
                        points[i],
                        camera_pose,
                        camera_rotation,
                        fovx,
                        width,
                        height,
                        bound=1.0 / 9,
                    )
                    - target_points[i]
                )
                for i in range(4)
            ]
        )

        return total_error

    from scipy.optimize import minimize

    roll_init = 0
    pitch_init = 0
    fovy_init = (fovy_range[0] + fovy_range[1]) / 2

    bounds = [
        (fovy_range[0], fovy_range[1]),
        (-np.pi / 90, np.pi / 90),
        (-np.pi / 2, 0),
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


# In the future, this function may be moved to scene_graph.py
def auto_get_optimal_camera_pose_for_object(
    view="top",  # 'top', 'human_full', 'human_focus'
    camera_xy=[0, 0],
    z_range=[0, 2.5],
    object_bbox=None,
    platform_rect=None,
    width=1920,
    height=1080,
    focus_ratio=0.5,
):

    key_points = None
    if view == "top":
        key_points = [object_bbox[0], object_bbox[3], object_bbox[1], object_bbox[2]]
    elif view == "human_full":
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

    #        if np.abs(np.cross(object_bbox[3] - object_bbox[0], object_bbox[5] - object_bbox[0])) > np.abs(np.cross(np.array(key_points[1]) - np.array(key_points[0]), np.array(key_points[3]) - np.array(key_points[0]))):
    #           key_points = [object_bbox[0], object_bbox[3], object_bbox[5], object_bbox[6]]
    else:
        raise ValueError("Invalid view type")

    optimal_error = 1e9
    optimal_z, optimal_roll, optimal_pitch, optimal_fovy = None, None, None, None

    for z in np.linspace(z_range[0], z_range[1], 10):
        for fovy_block in np.linspace(np.deg2rad(5), np.deg2rad(100), 1):
            camera_xyz = [camera_xy[0], camera_xy[1], z]
            [fovy, roll, pitch], error = solve_camera_pose_for_4_points(
                key_points,
                camera_xyz,
                width,
                height,
                focus_ratio,
                fovy_range=[fovy_block, fovy_block + np.deg2rad(95) / 1],
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

    fovy_32 = 2 * np.arctan(np.tan(optimal_fovy / 2) / focus_ratio)

    optimal_pose = sapien.Pose(
        p=[camera_xy[0], camera_xy[1], optimal_z],
        q=transforms3d.euler.euler2quat(
            optimal_roll, optimal_pitch, fixed_yaw, axes="sxyz"
        ),
    )
    print(
        "Optimal camera pose:",
        optimal_pose,
        "Optimal fovy:",
        np.rad2deg(optimal_fovy),
        "Optimal err:",
        optimal_error,
    )
    return optimal_pose, fovy_32


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
        cv2.fillPoly(img_array, [pts], color)

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGBA)

        return img_array

    # 2. Define a function to fill a rectangle (face of the cuboid)
    def fill_rectangle(rect_points, img_array, color, rectangle_grey):
        """Fills a rectangle defined by rect_points on the image array."""
        # a. Find bounding box of the rectangle
        min_x = int(np.min([p[0] for p in rect_points]))
        max_x = int(np.max([p[0] for p in rect_points]))
        min_y = int(np.min([p[1] for p in rect_points]))
        max_y = int(np.max([p[1] for p in rect_points]))

        # b. Clip to image boundaries
        min_x = max(0, min_x)
        max_x = min(width, max_x)
        min_y = max(0, min_y)
        max_y = min(height, max_y)

        if min_x >= max_x or min_y >= max_y:
            return img_array

        # c. Create grid of coordinates within the bounding box
        x, y = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
        x = x.flatten()
        y = y.flatten()

        # d. Use Shapely to determine which points are inside the polygon
        polygon = Polygon(rect_points)
        points = [Point(i, j) for i, j in zip(x, y)]
        mask = np.array([polygon.contains(point) for point in points]).reshape(
            max_y - min_y, max_x - min_x
        )

        # e. Apply color to the pixels inside the rectangle
        img_array[min_y:max_y, min_x:max_x][mask, :3] = (
            color + img_array[min_y:max_y, min_x:max_x][mask, :3]
        ) // 2

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


def auto_render_image_refactored(
    scene,
    pose,
    fovy,
    width=1920,
    height=1080,
    might_mark_object_cuboid_list=[],
    might_mark_freespace_list=[],
    rectangle_grey=False,
    save_path="image.png",
):

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

    colors = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
    colors = [tuple([int(color[i] * 255) for i in range(3)]) for color in colors]

    font = ImageFont.truetype(
        "msmincho.ttc", 40 if len(might_mark_object_cuboid_list) <= 10 else 25
    )
    number_idx = 0
    for might_mark_cuboid in might_mark_object_cuboid_list:
        cuboid_image_points = [
            imagepoint.world_to_image(
                point,
                camera.get_global_pose().p,
                transforms3d.quaternions.quat2mat(camera.get_global_pose().q),
                camera.fovx,
                width,
                height,
            )
            for point in might_mark_cuboid
        ]
        # Check if all points are inside the image boundaries
        all_points_inside = all(
            0 <= point[0] < width and 0 <= point[1] < height
            for point in cuboid_image_points
        )

        if all_points_inside:
            # Process the cuboid further if needed
            color = (
                colors[
                    might_mark_object_cuboid_list.index(might_mark_cuboid) % len(colors)
                ]
                if not rectangle_grey
                else (160, 160, 160)
            )
            img = draw_cuboid_on_image(
                img, cuboid_image_points, height, width, color, rectangle_grey
            )

            mid_cube = np.mean(might_mark_cuboid, axis=0)
            bold_pixel = [(0, 0), (1, 0)]
            font = ImageFont.truetype("msmincho.ttc", int(40))
            draw = ImageDraw.Draw(img)
            for bold_offset in bold_pixel:
                draw.text(
                    tuple(
                        np.array(
                            imagepoint.world_to_image(
                                mid_cube,
                                camera.get_global_pose().p,
                                transforms3d.quaternions.quat2mat(
                                    camera.get_global_pose().q
                                ),
                                camera.fovx,
                                width,
                                height,
                            )
                        )
                        + np.array(bold_offset)
                    ),
                    circled_numbers[number_idx % len(circled_numbers)],
                    font=font,
                    fill="black",
                )
            number_idx += 1

    for might_mark_freespace in might_mark_freespace_list:
        freespace_image_points = [
            imagepoint.world_to_image(
                point,
                camera.get_global_pose().p,
                transforms3d.quaternions.quat2mat(camera.get_global_pose().q),
                camera.fovx,
                width,
                height,
            )
            for point in might_mark_freespace
        ]
        # Check if all points are inside the image boundaries
        all_points_inside = all(
            0 <= point[0] < width and 0 <= point[1] < height
            for point in freespace_image_points
        )

        if all_points_inside or True:
            # Process the cuboid further if needed
            rect_in_cube_list = [[0, 1, 2, 3]]
            for rect_in_cube in rect_in_cube_list:
                pixels = box_on_image(
                    height, width, [freespace_image_points[i] for i in rect_in_cube]
                )
                for pixel in pixels:
                    current_color = img.getpixel(pixel)
                    color = (
                        (160, 160, 160)
                        if rectangle_grey
                        else colors[number_idx % len(colors)]
                    )
                    new_color = tuple(
                        [current_color[j] // 2 + color[j] // 2 for j in range(3)]
                    )
                    if not rectangle_grey or (pixel[0] + pixel[1]) % 2 == 0:
                        img.putpixel(pixel, new_color)
            mid_freespace = np.mean(might_mark_freespace, axis=0)
            bold_pixel = [(0, 0), (1, 0)]

            font = ImageFont.truetype("msmincho.ttc", int(20))
            draw = ImageDraw.Draw(img)
            for bold_offset in bold_pixel:
                draw.text(
                    tuple(
                        np.array(
                            imagepoint.world_to_image(
                                mid_freespace,
                                camera.get_global_pose().p,
                                transforms3d.quaternions.quat2mat(
                                    camera.get_global_pose().q
                                ),
                                camera.fovx,
                                width,
                                height,
                            )
                        )
                        + np.array(bold_offset)
                    ),
                    circled_numbers[number_idx % len(circled_numbers)],
                    font=font,
                    fill="black",
                )
            number_idx += 1

    img.save(save_path)
    camera.disable()

    return img


def main():

    # create a scene
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)

    json_file_path = "input/replica_apt_0.json"
    load_objects_from_json(scene, json_file_path)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    rect_points = [
        [0.4294, -5.7615, 0.195],
        [0.4294, -5.6088, 0.195],
        [0.5350, -5.6088, 0.195],
        [0.5350, -5.7615, 0.195],
    ]

    fixed_x, fixed_y = 1.01, -5.4
    z_min, z_max = 0.165, 0.505
    fovy_min, fovy_max = np.deg2rad(20), np.deg2rad(129)
    width, height = 1920, 1080

    rect_points = [
        [0.3557, -5.8785, 1.105],
        [0.3557, -4.972, 1.105],
        [0.7357, -4.972, 1.105],
        [0.7357, -5.8785, 1.105],
    ]

    fixed_x, fixed_y = 1.01, -5.4
    z_min, z_max = 1.105, 1.380 + 0.1
    fovy_min, fovy_max = np.deg2rad(20), np.deg2rad(179)
    width, height = 1920, 1080

    best_yaw, best_pitch, best_z, best_fovy = find_optimal_camera_pose(
        scene,
        rect_points=rect_points,
        fixed_x=fixed_x,
        fixed_y=fixed_y,
        z_min=z_min,
        z_max=z_max,
        fovy_min=fovy_min,
        fovy_max=fovy_max,
        width=width,
        height=height,
        absolute_precision=1e-2,
        rad_precision=1e-4,
        use_ternary_search=True,
    )

    # best_fovy = 2 * np.arctan(np.tan(best_fovx / 2) * height / width)

    camera = create_and_mount_camera(
        scene,
        pose=sapien.Pose(
            p=[fixed_x, fixed_y, best_z],
            q=transforms3d.euler.euler2quat(0, best_pitch, best_yaw, axes="sxyz"),
        ),
        near=0.1,
        far=1000,
        width=width,
        height=height,
        fovy=best_fovy,
        camera_name="camera",
    )

    img = render_image(scene, camera)
    img.save("debug_picture_taken.png")


"""
    draw_top_view_images_from_scene_graph(scene, 'input/replica_apt_0.txt', save_path='output')

    camera = create_and_mount_camera(scene, 
                                     pose=sapien.Pose(p=[0.4144, 0.1747, 2], q=[0.5, -0.5, 0.5, 0.5]),
                                     near=0.1,
                                     far=1000,
                                     width=1920,
                                     height=1080,
                                     fovy=np.deg2rad(90),
                                     camera_name='camera')
    
    img = render_image(scene, camera)
    
    camera_rotation_matrix = transforms3d.quaternions.quat2mat(camera.get_global_pose().q)
    camera_point = imagepoint.world_to_camera([1.11, 0.65197, 0.74835], camera.get_global_pose().p, camera_rotation_matrix )

    rotation_matrix_offset = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0 ,-1, 0]
    ])
    camera_point = np.dot(rotation_matrix_offset, camera_point)
    print(camera_point, camera.fovx)
    x,y = imagepoint.camera_to_image(camera_point, camera.fovx, 1920, 1080)
    print(x,y)

    img_array = np.array(img)
    img_array[y - 5:y + 5, x - 5:x + 5] = [255, 0, 0, 255]
    img = Image.fromarray(img_array)

    

    camera_point = imagepoint.world_to_camera([1.11,0.6006,0.74835], camera.get_global_pose().p, camera_rotation_matrix )
    camera_point = np.dot(rotation_matrix_offset, camera_point)
    print(camera_point)
    x,y = imagepoint.camera_to_image(camera_point, camera.fovx, 1920, 1080)
    print(x,y)

    img_array = np.array(img)
    img_array[y - 5:y + 5, x - 5:x + 5] = [255, 0, 0, 255]
    img = Image.fromarray(img_array)

    camera_point = imagepoint.world_to_camera([-0.2852,0.6006,0.74835], camera.get_global_pose().p, camera_rotation_matrix )
    camera_point = np.dot(rotation_matrix_offset, camera_point)
    print(camera_point)
    x,y = imagepoint.camera_to_image(camera_point, camera.fovx, 1920, 1080)
    print(x,y)

    img_array = np.array(img)
    img_array[y - 5:y + 5, x - 5:x + 5] = [255, 0, 0, 255]
    img = Image.fromarray(img_array)

    camera_point = imagepoint.world_to_camera([-0.2852,0.6519,0.74835], camera.get_global_pose().p, camera_rotation_matrix )
    camera_point = np.dot(rotation_matrix_offset, camera_point)
    print(camera_point)
    x,y = imagepoint.camera_to_image(camera_point, camera.fovx, 1920, 1080)
    print(x,y)

    img_array = np.array(img)
    img_array[y - 5:y + 5, x - 5:x + 5] = [255, 0, 0, 255]
    img = Image.fromarray(img_array)
    img.save('output/image_dot.png')

"""

# load objects from json


if __name__ == "__main__":
    main()
