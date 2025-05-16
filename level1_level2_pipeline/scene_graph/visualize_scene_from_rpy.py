import sapien as sapien
from sapien.utils import Viewer
import numpy as np
import json
import transforms3d
import json
import numpy as np
import transforms3d
import os
from pathlib import Path
import os.path as osp


def load_objects_from_json(scene, json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # load background
    bg_pose = sapien.Pose(q=q)
    bg_path = data["background_file_path"]

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(bg_path)
    builder.add_nonconvex_collision_from_file(bg_path)
    bg = builder.build_static(name=f"scene_background")
    bg.set_pose(bg_pose)

    # load objects
    for obj in data["object_instances"]:

        if obj["name"] == "GROUND":
            continue

        if (
            obj["name"] != "frl_apartment_rug_01"
            and obj["name"] != "frl_apartment_table_02"
        ):
            continue

        object_file_path = obj["glb_path"]
        position = [
            obj["centroid_translation"]["x"],
            obj["centroid_translation"]["y"],
            obj["centroid_translation"]["z"],
        ]

        rpy = []

        if obj["name"] == "frl_apartment_rug_01":
            rpy = [90.0, 0.0, 54.21796422]
        elif obj["name"] == "frl_apartment_table_02":
            rpy = [90.0, 0.0, 144.25753212]

        quaternion = transforms3d.euler.euler2quat(*np.deg2rad(rpy))
        quaternion = transforms3d.quaternions.qmult(q, quaternion)
        # quaternion = q * quaternion
        print(quaternion)

        builder = scene.create_actor_builder()
        builder.add_convex_collision_from_file(filename=object_file_path)
        builder.add_visual_from_file(filename=object_file_path)

        mesh = builder.build_static(name=obj["name"])
        mesh.set_pose(sapien.Pose(p=position, q=quaternion))

        print(obj["name"], obj["centroid_translation"], obj["quaternion"])

    print(builder.__dict__)


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


def reget_entities_from_sapien(scene, path="entities.json"):
    def convert_to_float(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_float(i) for i in obj]
        return obj

    res = {}
    for entity in scene.entities:
        res[entity.get_name() + "_" + str(entity.get_global_id())] = {}

        p = {
            "x": entity.get_pose().p[0],
            "y": entity.get_pose().p[1],
            "z": entity.get_pose().p[2],
        }
        q = {
            "w": entity.get_pose().q[0],
            "x": entity.get_pose().q[1],
            "y": entity.get_pose().q[2],
            "z": entity.get_pose().q[3],
        }
        res[entity.get_name() + "_" + str(entity.get_global_id())]["pose"] = (
            convert_to_float(p)
        )
        res[entity.get_name() + "_" + str(entity.get_global_id())]["quaternion"] = (
            convert_to_float(q)
        )

        res[entity.get_name() + "_" + str(entity.get_global_id())][
            "name"
        ] = entity.get_name()
        res[entity.get_name() + "_" + str(entity.get_global_id())][
            "id"
        ] = entity.get_global_id()

        print(entity.get_name(), entity.get_global_id())

    with open(path, "w") as f:
        json.dump(res, f, indent=4)


def main():

    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)

    # load objects from JSON
    json_file_path = "replica_apt_0.json"
    load_objects_from_json(scene, json_file_path)

    # set up lighting
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # set up viewer
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-5, y=0, z=6)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    # start simulation
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()

    reget_entities_from_sapien(scene)


if __name__ == "__main__":
    main()
