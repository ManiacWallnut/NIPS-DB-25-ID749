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
output_json_path = f"{current_path}/parsed_scene.json"
accurate_entity_path = f"{current_path}/parsed_scene.json"
#output_json_path = f"{current_path}/replica_apt_0.json"
#accurate_entity_path = f"{current_path}/replicaCAD_entities_apt_0.json"

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

visualize_ai2thor_sapien.load_objects_from_json(scene, output_json_path)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

for i in range(1000):
    scene.step()
    scene.update_render()


# %%
# This is only for debug, reuse the atomic_task.pkl to avoid the scene_graph_generation.
scene_graph_tree = None
scene_graph_tree_file_pkl = "scene_graph_tree_ai2thor.pkl"
task_sample_file_pkl = "task_sample_ai2thor.pkl"

if os.path.exists(scene_graph_tree_file_pkl) and True:
    with open(scene_graph_tree_file_pkl, "rb") as f:
        scene_graph_tree = pickle.load(f)
else:
    # 3: reading glbs,  generate scene graph
    ts = time.perf_counter()
    json_tree_path = gen_scene_graph.load_json_file(accurate_entity_path)
    scene_graph_tree = gen_scene_graph.gen_multi_layer_graph_with_free_space(
        json_tree_path
    )
    glog.info(f"scene graph tree generation time:  {time.perf_counter() - ts}")

with open(scene_graph_tree_file_pkl, "wb") as f:
    scene_graph_tree.corresponding_scene = None
    pickle.dump(scene_graph_tree, f)


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

#scene_graph_tree.rename_all_features(rename_dict)

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


# with open(task_sample_file_pkl, "wb") as f:
#     atomic_task.scene_graph_tree.corresponding_scene = None
#     pickle.dump(atomic_task, f)


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

# The description is now stored at start_task_msg_buffer, which will be sent to VLM together with the first task's information.


task_cnt = {}
for task in atomic_task.tasks:
        if task.type not in task_cnt:
            task_cnt[task.type] = []
        task_cnt[task.type].append(task)
    
task_sample = {}
wanted_task_list = [1,4,5,7,8,15,16,18,19,25,27,28,31,32,33,42,45,48,50,58,52,54,59, 60,61,64,66,76,80,81,82,83,87,90,95,97,100,101,104,105,108,110,115,121,124]
task_list = []
random.seed(2025)
for task_type in task_cnt.keys():
    if task_type not in task_sample:
        task_sample[task_type] = []
    #glog.info(task_cnt[task.type])#
    random.seed(2025)
    task_random_sample = random.choices(task_cnt[task_type], k=25)
    for id, task in enumerate(task_random_sample):
        
        
        if (id + task.type.value * 25) in wanted_task_list:#
            
            task_list.append(task)
            print(task.__repr_rough__())
        
    #     scene = sapien.Scene()
    #     scene.set_timestep(1 / 100)
    #     scene.add_ground(altitude=0)

    #     visualize_ai2thor_sapien.load_objects_from_json(scene, output_json_path)
    #     scene.set_ambient_light([0.5, 0.5, 0.5])
    #     scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    #     scene_graph_tree.corresponding_scene = scene
    #     #if task.type != 0:
    #         #task.task_debug(scene_graph_tree, id= id)
    # # task_sample[task.type] = task_random_sample
    

    #     for j in range(1000):
    #         scene.step()
    #         scene.update_render()


task_cnt = {k: len(v) for k, v in task_cnt.items()}
glog.info(f"Task count: {task_cnt}")



start_task_msg_buffer = ""

'''
Code below is for ReplicaCAD selecting tasks.
random.seed(2025)
scene_graph_tree.update_platform_children()
scene_graph_tree.corresponding_scene = None
initial_atomic_task = copy.deepcopy(atomic_task)
initial_scene_graph_tree = copy.deepcopy(scene_graph_tree)
task_sample = random.sample(atomic_task.tasks, 1800)
task_sample_ex = atomic_task.tasks
task_sample = task_sample + task_sample_ex
scene_graph_tree.corresponding_scene = scene
'''# import ipdb
# ipdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="online")
parser.add_argument("--model_name", type=str, default="llama-3-3b-70b-instruct")
parser.add_argument("--generate_reflection", type=bool, default=False)
parser.add_argument("--use_mistake_note", type=int, default=0)
args = parser.parse_args()
result = []
histories = []


task_type_cnt = [0 for i in range(5)]
total_score = 0
total_sr = 0


# correct_task_id_list = [6,9,16,19,26,31,34,38,45,51,63,69,93,94,129,163,173,209,251,279,287,450,742,920,1844,2580,2683,2957,3448,3561]
# result_file_path = f"{args.model_name}.txt"
# reflection_file_path = f"reflection.txt"

# if args.mode == 'generate_reflection':
#     with open(reflection_file_path, "w") as f:
#         f.write(f"To help you better finish the task, we provide you some tasks similar to the one you are given, and a correct sequence of actions and their explanations.\n")
# all_task_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 38, 43, 44, 45, 46, 51, 59, 62, 63, 67, 69, 77, 87, 93, 94, 95, 96, 112, 113, 119, 129, 163, 173, 185, 206, 209, 215, 219, 236, 251, 279, 280, 287, 300, 421, 450, 451, 481, 720, 742, 747, 754, 862, 866, 876, 920, 1176, 1844, 2460, 2580, 2683, 2918, 2957, 3448, 3561]
# wrong_task_id_list = [task_id for task_id in all_task_id_list if task_id not in correct_task_id_list]
# random.seed(2026)
# random.shuffle(wrong_task_id_list)
# wrong_task_id_train_list = wrong_task_id_list[:int(len(wrong_task_id_list) * 0.66)]
# wrong_task_id_test_list = wrong_task_id_list[int(len(wrong_task_id_list) * 0.66):]

# task_list = wrong_task_id_train_list if args.mode == 'generate_reflection' else wrong_task_id_test_list
result_file_path = f"{args.model_name}.txt"
reflection_file_path = f"reflection.txt"
all_task_id_list = []


scene_graph_tree.corresponding_scene = None
initial_atomic_task = copy.deepcopy(atomic_task)
initial_scene_graph_tree = copy.deepcopy(scene_graph_tree)
scene_graph_tree.corresponding_scene = scene
task_list = task_list[::-1]
import ipdb 
ipdb.set_trace()
for i, task in enumerate(task_list):
   
    task.generate_default_intermediate_state()
    task_type =  task.type.value
    task_type_cnt[task_type] += 1
    glog.info(f"Task {i}: {task.__repr_rough__()}")
    
   # all_task_id_list.append(i)
    
    
    # if args.mode == 'generate_reflection':
    #     goal_action_list, goal_explanation_list = task.generate_goal_information(scene_graph_tree)
    #     with open(reflection_file_path, "a") as f:
    #         f.write(f"Task {i}: {task.__repr_rough__()}\n. Here are the actions and explanations for this task:\n")
    #         action_explanation_pairs = list(zip(goal_action_list, goal_explanation_list))
    #         for i, (action, explanation) in enumerate(action_explanation_pairs):
    #             f.write(f"Action {i+1}/{len(action_explanation_pairs)}: {action}\n")
    #             f.write(f"Explanation {i+1}/{len(action_explanation_pairs)}: {explanation}\n")
        
        
        
        # continue
    
    
    # description of task has moved into apply function.
    manual_vlm_interactor = vlm_interactor.VLMInteractor(mode=args.mode)
    scene_graph_tree.corresponding_scene = scene
    scene_graph_tree.rename_all_features(rename_dict)
    
    glog.info(task.__repr_rough__())
    # return TaskStatusCode.SUCCESS or TaskStatusCode.FAILURE
    another_scene = sapien.Scene()
    another_scene.set_timestep(1 / 100)
    another_scene.add_ground(altitude=0)

    visualize_ai2thor_sapien.load_objects_from_json(another_scene, output_json_path)
    another_scene.set_ambient_light([0.5, 0.5, 0.5])
    another_scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    for j in range(1000):
        another_scene.step()
        another_scene.update_render()

    task = task_interact_helper.TaskInteractHelper(
        task=task,
        task_id=i,
        scene_graph_tree=scene_graph_tree,
        scene=another_scene,
        
        vlm_interactor=manual_vlm_interactor,
        img_path=f"{current_path}",
        model_name=args.model_name,  # 'gemini-1.5-flash', 'llama-3-3b-70b-instruct'
        reflection_file_path=reflection_file_path,
        generate_mistake_note=args.generate_reflection,
        use_mistake_note=args.use_mistake_note,
    )
    task.apply_action(state=task_interact_helper.InteractStates.NAVIGATION)
    result.append([task.status, task.partial_score])
    histories.append(task.action_history_list)
    scene = sapien.Scene()
    scene.set_timestep(1 / 100)
    scene.add_ground(altitude=0)

    visualize_ai2thor_sapien.load_objects_from_json(scene, output_json_path)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    for j in range(1000):
        scene.step()
        scene.update_render()

    scene_graph_tree = copy.deepcopy(initial_scene_graph_tree)
    atomic_task = copy.deepcopy(initial_atomic_task)

    # import ipdb; ipdb.set_trace()
    # start_task_msg_buffer = ""
    total_score += task.partial_score
    total_sr += int(task.status == True)
    with open(result_file_path, "a") as f:
        f.write(f"Task {i}: {task.status}, Task Type: level {task.task.is_ambiguous(scene_graph_tree) + 1}, type {task.task.type}, Score: {task.partial_score}\n")
        f.write(f"Task Info: {task.task.__repr_rough__()}\n")
        f.write(f"History: {task.action_history_list}\n")

with open(result_file_path, "a") as f:
    f.write(f"Total Score: {total_score  / np.sum(task_type_cnt)}\n")
    f.write(f"Total Success Rate: {total_sr / np.sum(task_type_cnt)}\n")
    f.write(f"Task Type Count: {task_type_cnt}\n")

