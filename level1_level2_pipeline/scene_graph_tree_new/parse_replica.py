# %%
import json
import numpy as np
import transforms3d
import os
from pathlib import Path
import os.path as osp
import trimesh
from scipy.spatial.transform import Rotation as R
import argparse

REPLICA_DATASET_ROOT_PATH = Path(
    "d:/workplace/scene_graph/task_generation/scene_graph/scene_datasets/replica_cad_dataset"
)

REPLICA_DATASET_ROOT_PATH = Path(
    "/home/weikang/Workspace/task-generation/scene_graph/scene_datasets/replica_cad_dataset"
)

REPLICA_DATASET_ROOT_PATH = Path(
    "/media/iaskbd/7107462746C3F786/workplace/task_generation/scene_graph/scene_datasets/replica_cad_dataset"
)


def get_glb_path(template_name):
    # object config

    obj_config_path = Path(
        f"d:/workplace/scene_graph/task_generation/scene_graph/scene_datasets/replica_cad_dataset/configs/objects/{template_name}.object_config.json"
    )
    obj_config_path = Path(
        f"{REPLICA_DATASET_ROOT_PATH}/configs/objects/{template_name}.object_config.json"
    )
    with open(obj_config_path, "r") as f:
        obj_config = json.load(f)

    # object glb file path from config
    relative_glb_path = obj_config["render_asset"]
    glb_file_path = os.path.normpath(obj_config_path.parent / relative_glb_path)
    return glb_file_path


def get_collision_path(template_name):

    obj_config_path = Path(
        f"d:/workplace/scene_graph/task_generation/scene_graph/scene_datasets/replica_cad_dataset/configs/objects/{template_name}.object_config.json"
    )
    obj_config_path = Path(
        f"{REPLICA_DATASET_ROOT_PATH}/configs/objects/{template_name}.object_config.json"
    )
    with open(obj_config_path, "r") as f:
        obj_config = json.load(f)

    if obj_config.get("collision_asset"):
        relative_collision_path = obj_config["collision_asset"]
        collision_file_path = os.path.normpath(
            obj_config_path.parent / relative_collision_path
        )
    else:
        collision_file_path = None
        assert (
            obj_config.get("use_bounding_box_for_collision")
            and obj_config["use_bounding_box_for_collision"]
        )
    return collision_file_path


def get_urdf_path(template_name):

    urdf_path = (
        REPLICA_DATASET_ROOT_PATH
        / "urdf"
        / f"{osp.basename(template_name)}/{osp.basename(template_name)}.urdf"
    )
    #  with open(urdf_path, 'r') as f:
    #       urdf_config = json.load(f)
    relative_urdf_path = (
        "../../urdf/"
        + f"{osp.basename(template_name)}/{osp.basename(template_name)}.urdf"
    )
    urdf_path = os.path.normpath(urdf_path.parent / relative_urdf_path)

    return urdf_path


def calculate_bbox(glb_path):
    # load glb
    mesh = trimesh.load(
        glb_path, force="scene"
    )  # force='scene' to load all meshes in the scene
    bbox_min, bbox_max = mesh.bounds

    # compute bbox
    bbox_size = bbox_max - bbox_min

    return {
        "x_length": bbox_size[0],
        "y_length": bbox_size[2],
        "z_length": bbox_size[1],
    }


# %%
def parse_replica(input_json_path, output_json_path):
    # load input JSON file
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # quaternion offset for 90 degree rotation around x-axis
    q_offset = transforms3d.quaternions.axangle2quat(
        np.array([1, 0, 0]), theta=np.deg2rad(90)
    )

    background_template_name = data["stage_instance"]["template_name"].split("/")[-1]
    bg_path = osp.join(
        REPLICA_DATASET_ROOT_PATH, f"stages/{background_template_name}.glb"
    )

    # output data
    # output_data = {"object_instances": []}
    output_data = {
        "background_file_path": bg_path,
        "object_instances": [],
        "articulate_instances": [],
    }
    """
    ground_obj = {
            "name": "GROUND",
            "motion_type": "STATIC",
            "glb_path": None,
            "centroid_translation": {
                "x": 0,
                "y": 0,
                "z": 0
                },
            "quaternion": {
                "w": 1,
                "x": 0,
                "y": 0,
                "z": 0
            },
            "bbox": {
                "x_length": 200,
                "y_length": 200,
                "z_length": 0
                }
        }
    output_data["object_instances"].append(ground_obj)
    """
    obj_idx = 0
    articulate_idx = 0

    desired_objects = []
    # desired_objects = ['frl_apartment_table_02']
    # desired_objects = ['frl_apartment_indoor_plant_02']
    # desired_objects = ['frl_apartment_beanbag']
    # desired_objects = ['frl_apartment_tvstand']
    # desired_objects = ['frl_apartment_kitchen_utensil_03', 'frl_apartment_kitchen_utensil_04']
    #  desired_objects = ['frl_apartment_bowl_07', 'frl_apartment_pan_01', 'frl_apartment_choppingboard_02', 'frl_apartment_kitchen_utensil_01', 'frl_apartment_kitchen_utensil_05', 'frl_apartment_lamp_02','frl_apartment_table_02' ]
    # desired_objects = ['frl_apartment_book_01','frl_apartment_book_02','frl_apartment_book_03','frl_apartment_book_04','frl_apartment_book_05','frl_apartment_wall_cabinet_02',]
    # desired_objects = ['frl_apartment_picture_02', 'frl_apartment_handbag', 'frl_apartment_shoebox']
    not_desired_objects = [
        "frl_apartment_handbag",
        "frl_apartment_cushion_01",
        "frl_apartment_monitor",
        "frl_apartment_cloth_01",
        "frl_apartment_cloth_02",
        "frl_apartment_cloth_03",
        "frl_apartment_cloth",
        "frl_apartment_umbrella",
        "frl_apartment_tv_screen",
        "frl_apartment_indoor_plant_01",
        "frl_apartment_monitor_stand",
        "frl_apartment_setupbox",
        "frl_apartment_beanbag",
        "frl_apartment_bike_01",
        "frl_apartment_bike_02",
        "frl_apartment_indoor_plant_02",
        "frl_apartment_picture_01",
        "frl_apartment_towel",
        "frl_apartment_rug_01",
        "frl_apartment_rug_02",
        "frl_apartment_rug_03",
        "frl_apartment_mat",
    ]
    if "_0" in input_json_path:
        not_desired_objects.extend(["frl_apartment_tv_object"])
    if "_1" in input_json_path:
        not_desired_objects.extend(
            [
                "frl_apartment_remote-control_01",
                "frl_apartment_cup_05",
                "frl_apartment_cup_03",
                "frl_apartment_bowl_01",
                "frl_apartment_bowl_02",
                "frl_apartment_bowl_03",
                "frl_apartment_vase_01",
                "frl_apartment_vase_02",
                "frl_apartment_camera_02",
            ]
        )

    if "_3" in input_json_path:
        not_desired_objects.extend(["frl_apartment_cup_03"])
    if "_4" in input_json_path:
        not_desired_objects.extend(
            [
                "frl_apartment_cup_02",
                "frl_apartment_cup_03",
                "frl_apartment_plate_01",
                "frl_apartment_vase_01",
                "frl_apartment_vase_02",
            ]
        )
    if "_5" in input_json_path:
        not_desired_objects.extend(
            [
                "frl_apartment_cup_03",
                "frl_apartment_vase_01",
                "frl_apartment_vase_02",
                "frl_apartment_bin_01",
            ]
        )
    # desired_articulations = ['kitchen_counter']
    # desired_articulations = ['kitchenCupboard_01']
    not_desired_articulations = ["kitchenCupboard_01"]
    desired_articulations = []
    glb = None
    for obj in data.get("object_instances", []):

        name = obj["template_name"].split("/")[-1]

        if len(desired_objects) and name not in desired_objects:
            continue

        if len(not_desired_objects) and name in not_desired_objects:
            continue

        glb_path = get_glb_path(name)
        collision_path = get_collision_path(name)
        glb = trimesh.load(glb_path)
        geometries = list(glb.geometry.values())
        # print(dir(glb.graph['frl_apartment_book_04']))

        node_name = next(key for key in glb.graph.nodes if key != "world")

        mat = glb.graph[node_name][0].copy()[:3, :3]
        # print(glb.graph[node_name][0])

        intrinsic_rotation = R.from_matrix(mat).as_quat()

        # print(node_name)
        """
        if 'frl_apartment_book_04' in glb.graph:
            print('graph',glb.graph['frl_apartment_book_04'],glb.graph.nodes)
            mat = glb.graph['frl_apartment_book_04'][0][:3,:3].copy()
            print(mat)
            print(R.from_matrix(mat).as_quat())
        """
        # print('vertice',geometries[0].vertices, 'bounds', geometries[0].bounds)

        transformed_vertices = trimesh.transform_points(
            geometries[0].vertices, glb.graph[node_name][0]
        )
        # print('transofrm',transformed_vertices,'bound',np.min(transformed_vertices, axis=0), np.max(transformed_vertices, axis=0))

        bbox_min, bbox_max = np.min(transformed_vertices, axis=0), np.max(
            transformed_vertices, axis=0
        )

        # print('correct bounds',bbox_min, bbox_max, glb.bounds)
        bbox = bbox_max - bbox_min
        bbox = {"x_length": bbox[0], "y_length": bbox[2], "z_length": bbox[1]}
        lowbound_file = "low_bounds.txt"
        # print(f"name: {name}_{obj_idx},bbox_move: {(bbox_min + bbox_max)  / 2},translation: {obj['translation']}", file=open(lowbound_file, "a"))
        translation_offset = (bbox_min + bbox_max) / 2

        motion_type = obj["motion_type"]
        if len(desired_objects) and name not in desired_objects:
            motion_type = "KEEP_FIXED"

        if len(not_desired_objects) and name in not_desired_objects:
            motion_type = "KEEP_FIXED"

        translation = obj["translation"]
        rotation = obj["rotation"]

        corrected_rotation = transforms3d.quaternions.qmult(q_offset, rotation)
        rpy = transforms3d.euler.quat2euler(corrected_rotation, axes="sxyz")
        corrected_rotation = transforms3d.euler.euler2quat(
            rpy[0] - np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
        )

        # Now the eps can be set into 0.017 rather than 0.2

        corrected_pose = {
            "centroid_translation": {
                "x": translation[0],  # +translation_offset[0],
                "y": -translation[2],  # +translation_offset[2],
                "z": translation[1],  # +translation_offset[1]
            },
            "quaternion": {
                "w": corrected_rotation[0],
                "x": corrected_rotation[1],
                "y": corrected_rotation[2],
                "z": corrected_rotation[3],
            },
        }

        # glb path

        # output object instance
        output_obj = {
            "name": f"{name}_{obj_idx}",
            "motion_type": motion_type,
            "visual_path": glb_path,
            "collision_path": collision_path,
            "centroid_translation": corrected_pose["centroid_translation"],
            "quaternion": corrected_pose["quaternion"],
            "bbox": bbox,
        }
        output_data["object_instances"].append(output_obj)
        obj_idx += 1

    for articulated_meta in data.get("articulated_object_instances", []):
        template_name = articulated_meta["template_name"]
        if len(desired_articulations) and template_name not in desired_articulations:
            continue
        if (
            len(not_desired_articulations)
            and template_name in not_desired_articulations
        ):
            continue
        # if 'door' in template_name:
        #     continue
        pos = articulated_meta["translation"]
        rotation = articulated_meta["rotation"]
        fixed_base = (
            articulated_meta["fixed_base"]
            if articulated_meta.get("fixed_base")
            else False
        )
        uniform_scale = (
            articulated_meta["uniform_scale"]
            if articulated_meta.get("uniform_scale")
            else 1.0
        )

        urdf_path = get_urdf_path(template_name)

        q_offset = transforms3d.quaternions.axangle2quat(
            np.array([1, 0, 0]), theta=np.deg2rad(90)
        )
        corrected_rotation = transforms3d.quaternions.qmult(q_offset, rotation)
        rpy = transforms3d.euler.quat2euler(corrected_rotation, axes="sxyz")
        corrected_rotation = transforms3d.euler.euler2quat(
            rpy[0] - np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
        )
        output_articulate = {
            "name": f"{template_name}_{articulate_idx}",
            "translation": {"x": pos[0], "y": -pos[2], "z": pos[1]},
            "urdf_path": urdf_path,
            "rotation": {
                "w": corrected_rotation[0],
                "x": corrected_rotation[1],
                "y": corrected_rotation[2],
                "z": corrected_rotation[3],
            },
            "fixed_base": fixed_base,
            "uniform_scale": uniform_scale,
        }
        output_data["articulate_instances"].append(output_articulate)
        articulate_idx += 1
    # save output JSON file

    if os.path.exists(output_json_path):
        os.remove(output_json_path)
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=4)
    import glog

    glog.info(f"output_json_path: {output_json_path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json_path",
        type=str,
        default="./scene_datasets/replica_cad_dataset/configs/scenes/apt_0.scene_instance.json",
    )
    parser.add_argument("--output_json_path", type=str, default="replica_apt_0.json")

    input_json_path = parser.parse_args().input_json_path
    output_json_path = parser.parse_args().output_json_path

    parse_replica(input_json_path, output_json_path)
    print("Output JSON file saved at:", output_json_path)


if __name__ == "__main__":
    # REPLICA_DATASET_ROOT_PATH = Path("../scene_graph/scene_datasets/replica_cad_dataset")

    main()
# %%
