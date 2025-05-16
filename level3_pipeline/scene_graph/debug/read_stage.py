
#%%
import sys 
import os 
from pygltflib import GLTF2, BufferFormat
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from typing import List
from shapely.geometry import Polygon, LinearRing, LineString, Point
import matplotlib.pyplot as plt
import copy
#%%

path_prefix = "/media/iaskbd-ubuntu/workplace/scene_graph/task_generation/scene_graph/"
#path_prefix = "D:/workplace/scene_graph/task_generation/scene_graph/"
stage_path = "scene_datasets/replica_cad_dataset/objects/frl_apartment_indoor_plant_01.glb"

glb = trimesh.load(os.path.join(path_prefix, stage_path))

#%%

path_prefix = "D:/workplace/scene_graph/task_generation/scene_graph/"


#%%

geometries = glb.geometry
#%%

single_mesh = glb.geometry['Mesh.575']
new_glb = trimesh.Trimesh(vertices=single_mesh.vertices, faces=single_mesh.faces)
new_glb.show()


#%%
anomesh = glb.geometry['Mesh.573']
new_glb = trimesh.Trimesh(vertices=anomesh.vertices, faces=anomesh.faces)
new_glb.apply_transform(np.array(glb.graph['frl_apartment_wall'][0]))
new_glb.show()

#%%
zero = np.zeros(shape=(4,4))
glb.geometry['Mesh.561'].apply_transform(zero)
glb.geometry['Mesh.575'].apply_transform(zero)
glb.geometry['Mesh.574'].apply_transform(zero)
glb.geometry['Mesh.566'].apply_transform(zero)
glb.geometry['Mesh.565'].apply_transform(zero)
glb.geometry['Mesh.564'].apply_transform(zero)
glb.geometry['Mesh.563'].apply_transform(zero)
glb.geometry['Mesh.547'].apply_transform(zero)
glb.geometry['Mesh.518'].apply_transform(zero)

#%%
'''
defaultdict(list,
            {'Mesh.575': ['frl_apartment_wall-plug_02'],
             'Mesh.574': ['frl_apartment_wall-plug_01'],
             'Mesh.573': ['frl_apartment_wall'],
             'Mesh.566': ['frl_apartment_switch_02'],
             'Mesh.565': ['frl_apartment_switch_01'],
             'Mesh.564': ['frl_apartment_stair'],
             'Mesh.563': ['frl_apartment_pipe'],
             'Mesh.562': ['frl_apartment_panel.001'],
             'Mesh.561': ['frl_apartment_floor'],
             'Mesh.560': ['frl_apartment_door_part.004'],
             'Mesh.555': ['frl_apartment_door_part.003'],
             'Mesh.554': ['frl_apartment_door_part.002'],
             'Mesh.553': ['frl_apartment_door_part.001'],
             'Mesh.552': ['frl_apartment_door_part'],
             'Mesh.551': ['frl_apartment_door_06'],
             'Mesh.550': ['frl_apartment_door_05'],
             'Mesh.549': ['frl_apartment_door_04'],
             'Mesh.548': ['frl_apartment_door_01'],
             'Mesh.547': ['frl_apartment_camera1'],
             'Mesh.518': ['frl_apartment_blinds']})

'''

#%%
    
# %%
for mesh_name, geometry in glb.geometry.items():
    if '573' in mesh_name or '518' in mesh_name or '548' in mesh_name or '549' in mesh_name or '550' in mesh_name or '551' in mesh_name:
            continue
    glb.geometry[mesh_name].apply_transform(zero)