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
import argparse
import xml.etree.ElementTree as ET
from .replica_urdf_adjustment import urdf_adjustment
from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch import (
    FETCH_BASE_COLLISION_BIT,
    FETCH_WHEELS_COLLISION_BIT,
)
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.structs import Actor, Articulation


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


fixed_objects = set()


def remove_objects(scene, object_names):
    for entity in scene.entities:
        if entity.get_name() in object_names:
            entity.remove_from_scene()


def add_objects(scene, obj):
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

    if "cushion_03" in object_file_path:
        position[2] += 0.1

    rpy = transforms3d.euler.quat2euler(quaternion, axes="sxyz")
    quaternion = transforms3d.euler.euler2quat(
        rpy[0] + np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
    )

    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial()
    material.set_base_color([1, 1, 1])
    material.set_metallic(0.0)
    material.set_roughness(0.5)
    material.set_specular(0.5)
    material.set_transmission(9.0)
    import ipdb

    ipdb.set_trace()
    builder.add_visual_from_file(filename=object_file_path, material=material)
    if collision_path is not None:
        builder.add_multiple_convex_collisions_from_file(filename=collision_path)
    else:

        builder.add_convex_collision_from_file(filename=object_file_path)

    if obj["motion_type"] == "STATIC" or obj["motion_type"] == "KEEP_FIXED":
        mesh = builder.build_static(name=obj["name"])
    else:
        mesh = builder.build(name=obj["name"])
    mesh.set_pose(sapien.Pose(p=position, q=quaternion))


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

        if "name" in obj and obj["name"] == "GROUND":
            continue
        if "motion_type" in obj and obj["motion_type"] == "KEEP_FIXED":
            fixed_objects.add(obj["name"])

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

        if "cushion_03" in object_file_path:
            position[2] += 0.1

        rpy = transforms3d.euler.quat2euler(quaternion, axes="sxyz")
        quaternion = transforms3d.euler.euler2quat(
            rpy[0] + np.deg2rad(90), rpy[1], rpy[2], axes="sxyz"
        )

        builder = scene.create_actor_builder()
        material = sapien.render.RenderMaterial()
        #   material.ior = 1.2
        material.transmission = 1e9
        material.base_color = [1.0, 1.0, 1.0, 1.0]
        builder.add_visual_from_file(filename=object_file_path)

        #     builder.add_visual_from_file(filename=object_file_path)
        if collision_path is not None:
            builder.add_multiple_convex_collisions_from_file(filename=collision_path)
        else:

            builder.add_convex_collision_from_file(filename=object_file_path)

        if obj["motion_type"] == "STATIC" or obj["motion_type"] == "KEEP_FIXED":
            mesh = builder.build_static(name=obj["name"])
        else:
            mesh = builder.build(name=obj["name"])
        mesh.set_pose(sapien.Pose(p=position, q=quaternion))

    for articulated_meta in data.get("articulate_instances", []):
        template_name = articulated_meta["name"]
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

            #      articulated_meta['uniform_scale'] = np.array([articulated_meta['uniform_scale'] for _ in range(3)])
            urdf_loader.scale = urdf_loader.uniform_scale = articulated_meta[
                "uniform_scale"
            ]

        #  print(articulation_name[:articulation_name.rfind('_')], pos[2])
        if articulation_name[: articulation_name.rfind("_")] in urdf_adjustment:
            pos[2] -= (
                urdf_adjustment[articulation_name[: articulation_name.rfind("_")]]
                - 0.005
            )
        #      print(articulation_name[:articulation_name.rfind('_')], pos[2])
        builder = urdf_loader.parse(urdf_path)[0][0]
        pose = sapien.Pose(pos, quaternion)
        builder.initial_pose = pose
        articulation = builder.build()


def load_articulations(scene):
    q = transforms3d.quaternions.axangle2quat(np.array([1, 0, 0]), theta=np.deg2rad(90))

    articulations: Dict[str, Articulation] = dict()

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
    collision_path_prefix="scene_datasets/replica_cad_dataset/objects/",
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
            if filename in fixed_objects or "ground" in filename:
                continue
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

    for obj in data.get("object_instances", []):
        for obj_idx in range(len(res["object_instances"])):
            if res["object_instances"][obj_idx]["name"] == obj["name"]:
                res["object_instances"][obj_idx]["motion_type"] = obj["motion_type"]
                res["object_instances"][obj_idx]["collision_path"] = obj[
                    "collision_path"
                ]
                res["object_instances"][obj_idx]["visual_path"] = obj["visual_path"]
                break

    with open(path, "w") as f:
        json.dump(res, f, indent=4)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file_path", type=str, default="replica_apt_1.json")
    parser.add_argument("--output_pose_path", type=str, default="entities_apt_1.json")

    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)

    # load objects from JSON
    json_file_path = parser.parse_args().json_file_path
    load_objects_from_json(scene, json_file_path)

    # set up lighting
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    scene.add_point_light([1.989, -5.822, 1], [0.5, 0.5, 0.5])
    """
    scene.add_point_light([1.2931, -5.7490, 1.0273],[1, 1, 1])
    scene.add_point_light([2.2649, -6.4652, 1.0273],[1, 1, 1])
    scene.add_point_light([2.6857, -5.8942, 1.0273],[1, 1, 1])
    scene.add_point_light([1.7139, -5.1780, 1.0273],[1, 1, 1])
    """
    scene.add_point_light([2 + 0.15, -5.75 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([1.2 + 0.15, -5.75 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([1.6 + 0.15, -6.5 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([1.6 + 0.15, -5 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([2 + 0.15, -6.5 - 0.15, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([2 + 0.15, -5.0 - 0.15, 1.0], [0.4, 0.4, 0.4])

    scene.add_point_light([2 - 0.1, -6.35, 1.0], [0.4, 0.4, 0.4])
    scene.add_point_light([2 + 0.1, -6.35, 1.0], [0.4, 0.4, 0.4])
    #  scene.add_point_light([1.2,-6.5,1.5],[2,2,2])
    scene.add_point_light([1.2 + 0.15, -5 - 0.15, 1.0], [0.4, 0.4, 0.4])

    # set up viewer
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-5, y=0, z=6)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    for entity in scene.entities:
        if "book_03" in entity.get_name():
            continue
        for component in entity.get_components():
            if isinstance(component, sapien.pysapien.render.RenderBodyComponent):
                component.shading_mode = 1
        #      component.visibility = 0.2

    #   scene.entities[4].get_components()[0].set_rendering_mode(
    #       sapien.render.RenderingMode.TRANSLUCENT
    #   )
    #  print(dir(scene.entities[1]))
    # import ipdb
    # ipdb.set_trace()

    # start simulation
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()

    reget_entities_from_sapien(
        scene, json_file_path, path=parser.parse_args().output_pose_path
    )


if __name__ == "__main__":
    main()
