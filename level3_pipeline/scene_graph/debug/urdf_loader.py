#%%
import xml.etree.ElementTree as ET
import trimesh
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
def parse_urdf_for_collision_planes(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    collision_planes = []

    for collision in root.findall(".//collision"):
        geometry = collision.find("geometry")
        box = geometry.find("box")
        mesh = geometry.find("mesh")
        print(collision, dir(collision))
        if box is not None:
            size = box.get("size")
            origin = collision.find("origin")
            if origin is not None:
                xyz = origin.get("xyz")
                rpy = origin.get("rpy")
                mesh_filename = mesh.get("filename") if mesh is not None else "N/A"
                collision_planes.append({
                    "size": size,
                    "xyz": xyz,
                    "rpy": rpy,
                    "mesh_filename": mesh_filename
                })

    return collision_planes

def parse_urdf_mesh(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    path_prefix = "D:/workplace/scene_graph/task_generation/scene_graph/"
    glb = trimesh.load(os.path.join(path_prefix, "scene_datasets/replica_cad_dataset/objects/frl_apartment_tvstand.glb"))
    geometries = []
    
    for link in root.findall(".//link"):
        visual = link.find("visual")
        if (visual is not None):
            origin = visual.find("origin")
            xyz = origin.get("xyz").split(" ")
            rpy = origin.get("rpy").split(" ")
            
            rpy = list(map(float, rpy))
            xyz = list(map(float, xyz))
            
            rotation = R.from_euler('xyz', rpy)
            rotation_matrix = rotation.as_matrix()
            
            transform_matrix = np.array(xyz)
            
            
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh")
            
            if mesh is not None:
                print(mesh.get("filename"))
                urdf_dir = os.path.dirname(urdf_path)
                mesh_path = os.path.join(urdf_dir, mesh.get("filename"))
                glb = trimesh.load(mesh_path)
                
                for k, v in glb.geometry.items():
                    
                    transformed_vertices = (rotation_matrix @ v.vertices.T).T + transform_matrix
                    
                    new_trimesh = trimesh.Trimesh(vertices=transformed_vertices, faces=v.faces)
                    geometries.append(new_trimesh)
    

# 示例使用
urdf_path = "d:/workplace/scene_graph/task_generation/scene_graph/scene_datasets/replica_cad_dataset/urdf/kitchen_counter/kitchenCupboard_01.urdf"
collision_planes = parse_urdf_for_collision_planes(urdf_path)

for plane in collision_planes:
    print(f"Size: {plane['size']}, Position: {plane['xyz']}, Rotation: {plane['rpy']}, mesh_filename: {plane['mesh_filename']}")
    
    
#%%
parse_urdf_mesh(urdf_path)

# %%
