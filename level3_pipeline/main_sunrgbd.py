# %%
import os
import sys
import sapien
import argparse

scene = sapien.Scene()
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("d:/workplace/scene_graph/task_generation/")
from scene_graph_tree_new import (
    parse_replica,
    gen_scene_graph,
    visualize_scene_sapien,
    visualize_ai2thor_sapien,
    scene_graph_tree_utils,
)
from custom_geometry_helper_new.convex_hull_processor import ConvexHullProcessor_2d
from image_renderer import image_render_processor
from atomic_task_generation import atomic_task_generation, task_interact_helper
from vlm_interactor import vlm_interactor
from vlm_interactor.item_classifier import renaming_engine
import numpy as np

import pickle
import random
import time
from enum import Enum
import colorama
from colorama import Fore, Style
import glog

import copy
import json

"""
TODO:

rename logic modification: 



"""
# TODO:
glog.info("import success")

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
current_path = os.path.dirname(os.path.abspath(__file__))
input_json_path = f"{current_path}/scene_graph/scene_datasets/replica_cad_dataset/configs/scenes/apt_0.scene_instance.json"
#output_json_path = f"{current_path}/parsed_scene.json"
#accurate_entity_path = f"{current_path}/parsed_scene.json"
output_json_path = f"{current_path}/replica_apt_0.json"
accurate_entity_path = f"{current_path}/replicaCAD_entities_apt_0.json"

NINE_DIRECTIONS = [
    "rear",
    "rear-left",
    "left",
    "front-left",
    "front",
    "front-right",
    "right",
    "rear-right",
    "center",
]

width = 1366
height = 768


class TaskStatusCode(Enum):
    SUCCESS = 0
    FAILURE = 1


def show_all(scene_graph_tree, save_path=None):
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
    for side in range(0, 2, 2):
        for node in scene_graph_tree.nodes.values():
            if "/" in node.name:
                node.name = node.name.split("/")[-1]
            if node.depth == 1:
                pass
            else:
                lside = side
                if node.depth > 1:
                    # import ipdb

                    #   ipdb.set_trace()
                    img = node.auto_take_non_ground_object_picture(
                        scene_graph_tree.corresponding_scene,
                        view="human_focus",  # 'human_focus', 'human_full', 'top_focus', 'top_full'
                        mark_object=False,  # if True, mark all the object on the same platform with cuboid.
                        only_mark_itself=False,
                        mark_freespace=True,
                        diagonal_mode="old",  # 'old', 'new_largest_rect', 'new_all', 'new_combined_freespace'
                        need_afford_rect=None,  # If not none, only mark the freespaces with size larger than it.
                        standing_direction=lside,
                        width=width,
                        height=height,
                        focus_ratio=0.5,
                        save_path=f"{save_path}{node.name}.png",
                        #   save_path=f"{save_path}{node.name[:node.name.rfind('_')]}.png",
                    )

                    # print("ground_object", ground_object.name, "platform", platform.name, "camera_pose", camera_pose, "free_space_height", min(ground_object.free_space_height[1], platform.avl_height))


# %%
# 0: parse replica
glog.info(f"input_json_path: {input_json_path}")
glog.info(f"output_json_path: {output_json_path}")
# parse_replica.parse_replica(input_json_path, output_json_path)

# %%
# 1: load scene
# sapien.render.set_camera_shader_dir("rt")
glog.info("init success")

scene.set_timestep(1 / 100)

visualize_scene_sapien.load_objects_from_json(scene, output_json_path)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

for i in range(1000):
    scene.step()
    scene.update_render()


# %%
# This is only for debug, reuse the atomic_task.pkl to avoid the scene_graph_generation.
scene_graph_tree = None
scene_graph_tree_file_pkl = "scene_graph_tree_sunrgbd.pkl"
task_sample_file_pkl = "task_sample_sunrgbd.pkl"
accurate_entity_path = f"{current_path}/sunrgbd_format_obj.json"
if os.path.exists(scene_graph_tree_file_pkl) and False:
    with open(scene_graph_tree_file_pkl, "rb") as f:
        scene_graph_tree = pickle.load(f)
else:
    # 3: reading glbs,  generate scene graph
    ts = time.perf_counter()
    json_tree_path = gen_scene_graph.load_json_file(accurate_entity_path)
    scene_graph_tree = gen_scene_graph.gen_scene_graph_for_sunrgbd(
        json_tree_path
    )
    glog.info(f"scene graph tree generation time:  {time.perf_counter() - ts}")

with open(scene_graph_tree_file_pkl, "wb") as f:
    scene_graph_tree.corresponding_scene = None
    pickle.dump(scene_graph_tree, f)

import ipdb
ipdb.set_trace()
scene_graph_tree.corresponding_scene = scene


# %%
# for node_name, node in scene_graph_tree.nodes.items():
#     if node.depth == 1:
#         for platform_id, platform in enumerate(node.own_platform):
#             glog.info(platform.name)
#             for dir in range(0, 8, 2):
#                 glog.info(f"{dir} {node.freespace_is_standable(dir)}")
#                 glog.info(f"{dir} {node.get_standing_point(dir)}")
#                 glog.info(f"{dir} {node.freespace_is_visible(platform_id = platform_id, standing_direction=dir)}")
# exit(0)

# %%

# show_all(scene_graph_tree, save_path=f"{current_path}/vlm_interactor/image4debug/")
# exit(0)
item_rename_processor = renaming_engine.ItemClassifier()

rename_dict = {}
if True and os.path.exists("rename_dict_replica.json"):
    with open("rename_dict_replica.json", "r") as f:
        rename_dict = json.load(f)
else:
    rename_dict = item_rename_processor.classify(
        img_path_folder=f"{current_path}/vlm_interactor/image4classify_replica/",
    )

    with open("rename_dict_replica.json", "w") as f:
        json.dump(rename_dict, f)

scene_graph_tree.rename_all_features(rename_dict)


# scene_graph_tree_utils.visualize_tree(
#     scene_graph_tree, file=open(f"{current_path}/scene_graph_tree.txt", "w")
# )
# show_all(scene_graph_tree, save_path=f"{current_path}/vlm_interactor/image4debug/")


# %%


if os.path.exists(task_sample_file_pkl) and False:
    with open(task_sample_file_pkl, "rb") as f:
        atomic_task = pickle.load(f)
    #    tasks = atomic_task.generate_all_tasks()
else:
    atomic_task = atomic_task_generation.TaskGeneration(scene_graph_tree)
    atomic_task.generate_task_from_scene_graph_tree()


with open(task_sample_file_pkl, "wb") as f:
    atomic_task.scene_graph_tree.corresponding_scene = None
    pickle.dump(atomic_task, f)


#%% 
import ipdb 
ipdb.set_trace()


# %%
global_vlm_interactor = vlm_interactor.VLMInteractor(mode="manual")

# %%
"""
self,
                                        scene,
                                        view='human', #top
                                        standing_direction=0,
                                        need_mark_rectangle_list = [],
                                        same_color=False,
                                        width=1366,
                                        height=768,
                                        focus_ratio=0.8,    
                                        save_path=None):
"""

# %%

random.seed(2025)
# 8: generate task sample and get sample images
task_sample = atomic_task.tasks
task_sample_ids = [atomic_task.tasks.index(task) for task in task_sample]


# The description is now stored at start_task_msg_buffer, which will be sent to VLM together with the first task's information.



start_task_msg_buffer = ""

scene_graph_tree.update_platform_children()
scene_graph_tree.corresponding_scene = None
initial_atomic_task = copy.deepcopy(atomic_task)
initial_scene_graph_tree = copy.deepcopy(scene_graph_tree)
task_sample = random.sample(atomic_task.tasks, 1800)
task_sample_ex = atomic_task.tasks
task_sample = task_sample + task_sample_ex
scene_graph_tree.corresponding_scene = scene
# import ipdb
# ipdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="online")
parser.add_argument("--model_name", type=str, default="llama-3-3b-70b-instruct")
args = parser.parse_args()
result = []
histories = []


'''
level1_cnt = 0
for i in range(len(atomic_task.tasks)):
    
    if atomic_task.tasks[i].is_ambiguous(scene_graph_tree):
        continue
    glog.info(f"Task {i}: {atomic_task.tasks[i].__repr_rough__()}")
    level1_cnt += 1
glog.info(f"Total level 1 tasks: {level1_cnt}")
for i in range(len(task_sample)):
    task = task_sample[i]
    if task_sample[i].is_ambiguous(scene_graph_tree):
        continue
    task.task_debug(scene_graph_tree,id=i)
    scene_graph_tree = copy.deepcopy(initial_scene_graph_tree)
    
    scene = sapien.Scene()
    scene.set_timestep(1 / 100)
    scene.add_ground(altitude=0)

    visualize_scene_sapien.load_objects_from_json(scene, output_json_path)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    for j in range(1000):
        scene.step()
        scene.update_render()
    scene_graph_tree.corresponding_scene = scene
    scene_graph_tree.rename_all_features(rename_dict)
'''

task_type_cnt = [0 for i in range(10)]
total_score = 0
total_sr = 0


correct_task_id_list = [6,9,16,19,26,31,34,38,45,51,63,69,93,94,129,163,173,209,251,279,287,450,742,920,1844,2580,2683,2957,3448,3561]
result_file_path = f"{args.model_name}.txt"
reflection_file_path = f"reflection.txt"

if args.mode == 'generate_reflection':
    with open(reflection_file_path, "w") as f:
        f.write(f"To help you better finish the task, we provide you some tasks similar to the one you are given, and a correct sequence of actions and their explanations.\n")
all_task_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 38, 43, 44, 45, 46, 51, 59, 62, 63, 67, 69, 77, 87, 93, 94, 95, 96, 112, 113, 119, 129, 163, 173, 185, 206, 209, 215, 219, 236, 251, 279, 280, 287, 300, 421, 450, 451, 481, 720, 742, 747, 754, 862, 866, 876, 920, 1176, 1844, 2460, 2580, 2683, 2918, 2957, 3448, 3561]
wrong_task_id_list = [task_id for task_id in all_task_id_list if task_id not in correct_task_id_list]
random.seed(2026)
random.shuffle(wrong_task_id_list)
wrong_task_id_train_list = wrong_task_id_list[:int(len(wrong_task_id_list) * 0.66)]
wrong_task_id_test_list = wrong_task_id_list[int(len(wrong_task_id_list) * 0.66):]

full_id_list = [i for i in range(len(task_sample))]

task_list = wrong_task_id_train_list if args.mode == 'generate_reflection' else all_task_id_list
print(len(task_list))
#task_list = task_list[::-1]

level3_task_list = [(i, i+300) for i in range(0, 30)]



for i in range(24, 30):
    task1 = task_sample[level3_task_list[i][0]]
    
    for task2_id, task2 in enumerate(task_sample):
        if task2_id == task1:
            continue
        elif task2.item.name == task1.item.name and task1.__repr__() != task2.__repr__():
            level3_task_list[i] = (level3_task_list[i][0], task2_id)
            print(task1, task2)
            break
            


level3_task_list = [(0, 300), (1, 301), (2, 302), (3, 303), (4, 304), (5, 305), (6, 306), (118, 307), (8, 308), (9, 309), (10, 310), (11, 311), (12, 312), (13, 313), (14, 314), (15, 14), (16, 316), (17, 317), (18, 318), (19, 319), (20, 320), (21, 321), (22, 322), (23, 323), (24, 39), (25, 35), (26, 11), (27, 40), (28, 475), (29, 23)]
# import ipdb 
# ipdb.set_trace()

import reload_history

history_path = "/media/iaskbd/E470A7DC9B7152FB/workplace/task_generation/level3-gpt-4o.txt"
histories = reload_history.extract_histories(history_path)

x = 0
for i, j in level3_task_list:
    if i >= 9 and i <= 24 or i < 5:
        continue
    task1 = task_sample[i]
    task2 = task_sample[j]
    print(task1, 'and', task2)
    
    all_task_id_list.append((i,j))
    
    another_scene = sapien.Scene()
    another_scene.set_timestep(1 / 100)
    another_scene.add_ground(altitude=0)

    visualize_scene_sapien.load_objects_from_json(another_scene, output_json_path)
    another_scene.set_ambient_light([0.5, 0.5, 0.5])
    another_scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    for j1 in range(1000):
        
        another_scene.step()
        another_scene.update_render()
    
    
    # description of task has moved into apply function.
    manual_vlm_interactor = vlm_interactor.VLMInteractor(mode=args.mode, manual_list = histories[x])
    x += 1
    scene_graph_tree.corresponding_scene = another_scene
    scene_graph_tree.rename_all_features(rename_dict)
    scene_graph_tree.corresponding_scene = scene
    scene_graph_tree.rename_all_features(rename_dict)
    
    
    glog.info(task1.__repr_rough__())
    # return TaskStatusCode.SUCCESS or TaskStatusCode.FAILURE
    task = task_interact_helper.TaskInteractHelper(
        task=task2,
        intermediate_task=task1,
        task_id=i,
        intermediate_task_id=j,
        scene_graph_tree=scene_graph_tree,
        scene=another_scene,
        vlm_interactor=manual_vlm_interactor,
        img_path=f"{current_path}",
        model_name=args.model_name,  # 'gemini-1.5-flash', 'llama-3-3b-70b-instruct'
        reflection_file_path=reflection_file_path,
        generate_mistake_note=False,
        use_mistake_note=0,
    )
    task.apply_action(state=task_interact_helper.InteractStates.NAVIGATION)
    result.append([task.status, task.partial_score])
    histories.append(task.action_history_list)
    scene = sapien.Scene()
    scene.set_timestep(1 / 100)
    scene.add_ground(altitude=0)

    visualize_scene_sapien.load_objects_from_json(scene, output_json_path)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    for j1 in range(1000):
        scene.step()
        scene.update_render()

    scene_graph_tree = copy.deepcopy(initial_scene_graph_tree)
    atomic_task = copy.deepcopy(initial_atomic_task)

    # import ipdb; ipdb.set_trace()
    # start_task_msg_buffer = ""
    total_score += task.partial_score
    total_sr += int(task.status == True)
    with open(result_file_path, "a") as f:
        f.write(f"Task {i}: {task.status and task.intermediate_partial_score==4}, Task Type: level {task_sample[i].is_ambiguous(scene_graph_tree) + 1}, type {task_sample[i].type}, Score: {task.intermediate_partial_score} {task.partial_score}\n")
        f.write(f"Task Info: {task.task_description}\n")
        f.write(f"History: {task.action_history_list}\n")

with open(result_file_path, "a") as f:
    f.write(f"Total Score: {total_score  / np.sum(task_type_cnt)}\n")
    f.write(f"Total Success Rate: {total_sr / np.sum(task_type_cnt)}\n")
    f.write(f"Task Type Count: {task_type_cnt}\n")

