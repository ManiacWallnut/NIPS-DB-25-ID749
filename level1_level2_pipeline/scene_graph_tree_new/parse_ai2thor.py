# %%
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
from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch import (
    FETCH_BASE_COLLISION_BIT,
    FETCH_WHEELS_COLLISION_BIT,
)
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs import Actor, Articulation

# %%

ROOT_PATH = Path(os.getcwd()).resolve().parents[1]
DATASET_ROOT_PATH = os.path.join(ROOT_PATH, "ai2thor")
MAIN_CODE_PATH = os.path.join(ROOT_PATH, "task_generation")
OBJECT_ROOT_PATH = os.path.join(
    DATASET_ROOT_PATH, "ai2thorhab-uncompressed", "assets", "objects"
)


# %%


def rotation_matrix_to_euler_angles(R):
    """
    将3x3旋转矩阵转换为欧拉角（ZYX顺序，即yaw-pitch-roll）
    返回角度值（度）
    """
    # 检查是否存在万向锁问题
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # roll
        y = np.arctan2(-R[2, 0], sy)  # pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    # 转换为度
    return np.array([np.rad2deg(z), np.rad2deg(y), np.rad2deg(x)])


def compute_rotation_matrix(
    standard_up=np.array([0, 0, 1]),
    standard_front=np.array([0, 1, 0]),
    target_up=np.array([0, 0, 1]),
    target_front=np.array([0, 1, 0]),
):
    """
    计算从标准坐标系到目标坐标系的旋转矩阵

    参数:
    - standard_up: 标准坐标系的up向量
    - standard_front: 标准坐标系的front向量
    - target_up: 目标坐标系的up向量
    - target_front: 目标坐标系的front向量

    返回:
    - 3x3旋转矩阵
    """
    # 将输入转换为 numpy 数组并归一化
    standard_up = np.array(standard_up, dtype=float)
    standard_front = np.array(standard_front, dtype=float)
    target_up = np.array(target_up, dtype=float)
    target_front = np.array(target_front, dtype=float)

    standard_up = standard_up / np.linalg.norm(standard_up)
    standard_front = standard_front / np.linalg.norm(standard_front)
    target_up = target_up / np.linalg.norm(target_up)
    target_front = target_front / np.linalg.norm(target_front)

    # 使用右手定则计算标准坐标系的第三个方向（right）
    standard_right = np.cross(standard_front, standard_up)
    standard_right = standard_right / np.linalg.norm(standard_right)

    # 确保standard_front是正交的（可能输入不完全正交）
    standard_front = np.cross(standard_up, standard_right)
    standard_front = standard_front / np.linalg.norm(standard_front)

    # 使用右手定则计算目标坐标系的第三个方向（right）
    target_right = np.cross(target_front, target_up)
    target_right = target_right / np.linalg.norm(target_right)

    # 确保target_front是正交的
    target_front = np.cross(target_up, target_right)
    target_front = target_front / np.linalg.norm(target_front)

    # 构建标准坐标系的基矩阵
    standard_basis = np.column_stack([standard_right, standard_front, standard_up])

    # 构建目标坐标系的基矩阵
    target_basis = np.column_stack([target_right, target_front, target_up])

    # 计算旋转矩阵：R = target_basis * standard_basis^(-1)
    rotation_matrix = np.dot(target_basis, np.linalg.inv(standard_basis))

    return rotation_matrix


# %%


def get_absolute_asset_path(json_file_path):
    """
    从JSON文件中获取render_asset的绝对路径

    参数:
        json_file_path: JSON文件的路径

    返回:
        render_asset的绝对路径
    """
    # 获取JSON文件的目录
    json_dir = os.path.dirname(os.path.abspath(json_file_path))

    # 读取JSON文件
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # 获取render_asset的相对路径
    relative_path = data.get("render_asset")

    if not relative_path:
        return None

    json_dir = json_dir.replace("ai2thor-hab", "ai2thorhab-uncompressed")
    # 将相对路径转换为绝对路径
    absolute_path = os.path.normpath(os.path.join(json_dir, relative_path))

    return absolute_path


def get_object_config_path(template_name):
    """
    Get the path to the glb config file for a given template name.
    """
    # Get the path to the glb config file
    object_config_path = os.path.join(
        DATASET_ROOT_PATH,
        "ai2thor-hab",
        "configs",
        template_name + ".object_config.json",
    )
    return object_config_path


def get_scene_glb_path(template_name):
    """
    Get the path to the glb file for a given scene template name.
    """
    import glog

    #    glog.info(f'Getting scene glb path for template: {template_name}')
    scene_glb_path = os.path.join(
        DATASET_ROOT_PATH, "ai2thor-hab", "assets", template_name + ".glb"
    )
    glog.info(f"Found scene glb path: {scene_glb_path} for template: {template_name}")
    return scene_glb_path


def parse_ai2thor(input_scene_json_path, output_scene_json_path):
    """
    Parse the AI2Thor scene JSON file and convert it to a ManiSkill-compatible format.

    Parameters:
    - input_scene_json_path: Path to the input AI2Thor scene JSON file.
    - output_scene_json_path: Path to save the output ManiSkill scene JSON file.
    """
    # Load the AI2Thor scene JSON file
    with open(input_scene_json_path, "r") as f:
        scene_data = json.load(f)

    output_json = {}
    background_glb = scene_data["stage_instance"]["template_name"]
    output_json["background_file_path"] = get_scene_glb_path(background_glb)

    instance_json_list = []
    for obj in scene_data["object_instances"]:
        template_name = obj["template_name"]
        object_config_path = get_object_config_path(template_name)
        object_config_data = {}
        if not os.path.exists(object_config_path):
            print(f"Object config file not found: {object_config_path}")
            continue
        # Load the object config file
        with open(object_config_path, "r") as f:
            object_config_data = json.load(f)

        object_glb_path = get_absolute_asset_path(object_config_path)
        up, front = np.array(object_config_data["up"]), np.array(
            object_config_data["front"]
        )

        # this rotation matrix rotates the current up and front vectors to the standard up and front vectors

        rotation_matrix = compute_rotation_matrix(
            standard_up=np.array([0, 0, 1]),
            standard_front=np.array([0, 1, 0]),
            target_up=up,
            target_front=front,
        )

        position = obj["translation"]
        rotation = obj["rotation"]

        if True or "Bathroom_Faucet_10" not in template_name:
            position = np.dot(np.linalg.inv(rotation_matrix), position)
            rotation_xyz = np.dot(rotation[-3:], rotation_matrix)
            rotation = np.concatenate((rotation[:1], rotation_xyz))

        non_uniform_scale = obj["non_uniform_scale"]

        # Get the glb path and config path for the object
        glb_config_path = get_object_config_path(template_name)

        # Create a dictionary to store the object data
        object_data = {
            "template_name": template_name,
            "glb_config_path": glb_config_path,
            "visual_path": object_glb_path,
            "collision_path": object_glb_path,  # No collision path in AI2Thor,
            "centroid_translation": {
                "x": position[0],
                "y": position[1],
                "z": position[2],
            },
            "quaternion": {
                "w": rotation[0],
                "x": rotation[1],
                "y": rotation[2],
                "z": rotation[3],
            },
            "non_uniform_scale": non_uniform_scale,
        }

        # Append the object data to the output JSON
        instance_json_list.append(object_data)

    output_json["object_instances"] = instance_json_list
    with open(output_scene_json_path, "w") as f:
        json.dump(output_json, f, indent=4)


def main():
    """
    ithor-floorplan4
    robothor-floorplan train2

    """
    parser = argparse.ArgumentParser(description="Parse AI2Thor scene JSON file.")
    # parser.add_argument(
    #     "--input_scene_json_path",
    #     type=str,
    #     default=os.path.join(DATASET_ROOT_PATH, "ai2thor-hab", "configs", "scenes", "iTHOR" ,"FloorPlan6_physics.scene_instance.json"),
    #     help="Path to the input AI2Thor scene JSON file.",
    # )
    parser.add_argument(
        "--input_scene_json_path",
        type=str,
        default=os.path.join(
            DATASET_ROOT_PATH,
            "ai2thor-hab",
            "configs",
            "scenes",
            "RoboTHOR",
            "FloorPlan_Train2_1.scene_instance.json",
        ),
        help="Path to the input AI2Thor scene JSON file.",
    )
    parser.add_argument(
        "--output_scene_json_path",
        type=str,
        default=os.path.join(MAIN_CODE_PATH, "parsed_scene.json"),
        help="Path to save the output ManiSkill scene JSON file.",
    )

    args = parser.parse_args()

    parse_ai2thor(args.input_scene_json_path, args.output_scene_json_path)
    print(f"Parsed scene JSON file saved to {args.output_scene_json_path}")


# %%
if __name__ == "__main__":
    main()

# %%
