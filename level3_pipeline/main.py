# %%
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("d:/workplace/scene_graph/task_generation/")
from scene_graph_tree_new import parse_replica, gen_scene_graph, visualize_scene_sapien
from custom_geometry_helper_new.convex_hull_processor import ConvexHullProcessor_2d
from image_renderer import image_render_processor
from atomic_task_generation import atomic_task_generation, task_interact_helper
from vlm_interactor import vlm_interactor
import numpy as np
import sapien
import pickle
import random
import time
from enum import Enum
import colorama
from colorama import Fore, Style
import glog


"""
TODO:

RENAME all the objects with the same catagory "kitchen utensils" . VLM will be called to do this.

After renaming, for level 1 tasks, don't render the objects at the beginning of the task; for level 2 task, render a focus view IN ALL DIRECTIONS.

Whenever the robot goto a platform, render "focus view" for ALL the objects on the platform. 

When generating tasks, generate task if it can be finished in any standing direction.



Show ALL the free spaces of ALL objects, and we'll check them one by one.

For level 1 tasks, try

FIX issues while rendering the freespaces, deal with out stretching lines.


"""
# TODO:


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
current_path = os.path.dirname(os.path.abspath(__file__))
input_json_path = f"{current_path}/scene_graph/scene_datasets/replica_cad_dataset/configs/scenes/apt_0.scene_instance.json"
output_json_path = f"{current_path}/scene_graph/replica_apt_0.json"
accurate_entity_path = f"{current_path}/scene_graph/entities_apt_0.json"

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
    for side in range(0, 8, 2):
        for node in scene_graph_tree.nodes.values():
            if node.depth == 1:
                if not node.freespace_is_standable(side):
                    continue
                for platform_id, platform in enumerate(node.own_platform):
                    if not node.freespace_is_visible(side, platform_id):
                        continue
                    img = node.auto_take_ground_object_picture(
                        scene_graph_tree.corresponding_scene,
                        view="human_full",
                        mark_object=len(platform.children) > 0,
                        mark_freespace=len(platform.children) == 0,
                        platform_id=platform_id,
                        standing_direction=side,
                        width=width,
                        height=height,
                        focus_ratio=0.75,
                        save_path=f"{save_path}{node.name}_{EIGHT_DIRECTIONS[side]}_{platform.name}.png",
                    )
                    img = node.auto_take_ground_object_picture(
                        scene_graph_tree.corresponding_scene,
                        view="top_full",
                        mark_object=len(platform.children) > 0,
                        mark_freespace=len(platform.children) == 0,
                        platform_id=platform_id,
                        standing_direction=side,
                        width=width,
                        height=height,
                        focus_ratio=0.75,
                        save_path=f"{save_path}{node.name}_{EIGHT_DIRECTIONS[side]}_{platform.name}.png",
                    )
            else:
                lside = side
                if node.depth > 1:
                    import ipdb

                    #   ipdb.set_trace()
                    img = node.auto_take_non_ground_object_picture(
                        scene_graph_tree.corresponding_scene,
                        view="human_focus",  # 'human_focus', 'human_full', 'top_focus', 'top_full'
                        mark_object=True,  # if True, mark all the object on the same platform with cuboid.
                        only_mark_itself=True,
                        mark_freespace=True,
                        diagonal_mode="old",  # 'old', 'new_largest_rect', 'new_all', 'new_combined_freespace'
                        need_afford_rect=None,  # If not none, only mark the freespaces with size larger than it.
                        standing_direction=lside,
                        width=width,
                        height=height,
                        focus_ratio=0.5,
                        save_path=f"{save_path}{node.name}_{EIGHT_DIRECTIONS[side]}_{node.name}_humanfocus.png",
                    )
                    img = node.auto_take_non_ground_object_picture(
                        scene_graph_tree.corresponding_scene,
                        view="top_focus",  # 'human_focus', 'human_full', 'top_focus', 'top_full'
                        mark_object=True,  # if True, mark all the object on the same platform with cuboid.
                        only_mark_itself=True,
                        mark_freespace=True,
                        diagonal_mode="old",  # 'old', 'new_largest_rect', 'new_all', 'new_combined_freespace'
                        need_afford_rect=None,  # If not none, only mark the freespaces with size larger than it.
                        standing_direction=lside,
                        width=width,
                        height=height,
                        focus_ratio=0.75,
                        save_path=f"{save_path}{node.name}_{EIGHT_DIRECTIONS[side]}_{node.name}_topfocus.png",
                    )
                    img = node.auto_take_non_ground_object_picture(
                        scene_graph_tree.corresponding_scene,
                        view="top_full",  # 'human_focus', 'human_full', 'top_focus', 'top_full'
                        mark_object=True,  # if True, mark all the object on the same platform with cuboid.
                        only_mark_itself=True,
                        mark_freespace=True,
                        diagonal_mode="old",  # 'old', 'new_largest_rect', 'new_all', 'new_combined_freespace'
                        need_afford_rect=None,  # If not none, only mark the freespaces with size larger than it.
                        standing_direction=lside,
                        width=width,
                        height=height,
                        focus_ratio=0.35,
                        save_path=f"{save_path}{node.name}_{EIGHT_DIRECTIONS[side]}_{node.name}_topfull.png",
                    )

                    """
                            
                            
                                def auto_take_non_ground_object_picture(self,
                                            scene,
                                            view='human_full', # 'human_focus', 'human_full', 'top_focus', 'top_full'
                                            mark_object=False, # if True, mark all the object on the same platform with cuboid.
                                            only_mark_itself=False, # if True, only mark itself
                                            mark_freespace=False,
                                            diagonal_mode='old', # 'old', 'new_largest_rect', 'new_all', 'new_combined_freespace'
                                            need_afford_rect=None, #If not none, only mark the freespaces with size larger than it.
                                            standing_direction=0,
                                            width=640,
                                            height=480,
                                            focus_ratio=0.8,
                                            save_path=None):
                                
                                def auto_take_ground_object_picture(self, 
                                        scene,
                                        view='human_full', # 'human_full', 'overall_top', 
                                        mark_object=False, 
                                        mark_freespace=False,
                                        need_afford_rect=None, #If not none, only mark the freespaces with size larger than it.
                                        platform_id=0, 
                                        standing_direction=0, 
                                        width=1366, 
                                        height=768, 
                                        focus_ratio=0.5,
                                        save_path=None):
                            """

                    # print("ground_object", ground_object.name, "platform", platform.name, "camera_pose", camera_pose, "free_space_height", min(ground_object.free_space_height[1], platform.avl_height))


def show_all_object_free_space(scene_graph_tree):
    for non_ground_object in scene_graph_tree.nodes.values():
        if non_ground_object.depth <= 1:
            continue
        belonged_ground_object = non_ground_object.parent
        while belonged_ground_object.depth > 1:
            belonged_ground_object = belonged_ground_object.parent

        for side in range(0, 8, 2):

            center = (
                belonged_ground_object.free_space[side]["Critical_space"][2]
                + belonged_ground_object.free_space[side]["Critical_space"][0]
            ) / 2

            if "lamp" in non_ground_object.name:
                print("lamp", non_ground_object.free_space)

            object_center = (
                non_ground_object.object.bbox[0] + non_ground_object.object.bbox[2]
            ) / 2
            object_top_center = np.array(
                [object_center[0], object_center[1], non_ground_object.bottom]
            )
            critical_free_spaces = [
                non_ground_object.free_space[i]["Critical_space"] for i in range(8)
            ]

            rect_points_for_optimize = [
                critical_free_spaces[1][0],
                critical_free_spaces[3][1],
                critical_free_spaces[5][2],
                critical_free_spaces[7][3],
            ]
            rect_points_for_optimize = [
                np.array([point[0], point[1], non_ground_object.bottom])
                for point in rect_points_for_optimize
            ]

            fixed_x, fixed_y = center[0], center[1]
            z_min, z_max = non_ground_object.free_space_height[0], min(
                non_ground_object.free_space_height[1], belonged_ground_object.top + 0.3
            )

            fovy_min, fovy_max = np.deg2rad(20), np.deg2rad(160)

            best_yaw, best_pitch, best_z, best_fovy = (
                image_render_processor.find_optimal_camera_pose(
                    scene,
                    rect_points=rect_points_for_optimize,
                    fixed_x=fixed_x,
                    fixed_y=fixed_y,
                    z_min=z_min,
                    z_max=z_max,
                    fovy_min=fovy_min,
                    fovy_max=fovy_max,
                    width=1366,
                    height=768,
                    absolute_precision=1e-3,
                    rad_precision=0.01,
                    use_ternary_search=True,
                )
            )
            camera_pose = np.array([center[0], center[1], best_z])
            print(
                "non_ground_object",
                non_ground_object.name,
                "camera_pose",
                camera_pose,
                "free_space_height",
                min(non_ground_object.free_space_height[1], belonged_ground_object.top),
            )
            if "book_01_29" in non_ground_object.name:
                print("book_01_29", non_ground_object.free_space)
            image_render_processor.draw_non_ground_object_views_on_image(
                scene,
                object_top_center=object_top_center,
                camera_pose=camera_pose,
                camera_rpy=[0, best_pitch, best_yaw],
                free_spaces=critical_free_spaces,
                save_path=f"d:/workplace/scene_graph/task_generation/scene_images/debug_image_save/{non_ground_object.name}_{gen_scene_graph.EIGHT_DIRECTIONS[side]}.png",
                width=1366,
                height=768,
                fovy=best_fovy,
            )

    pass


def give_vlm_scene_graph_tree_info(root_node="GROUND"):

    non_ground_objects = []
    ground_objects = []

    for node in scene_graph_tree.nodes.values():
        if node.depth == 1:
            ground_objects.append(node)
        elif node.depth > 1:
            non_ground_objects.append(node)

    send_str_buffer = (
        "Suppose you are a home robot. You're now in a room, and are given some kinds of tasks, which mainly involves puttings some thing from one place to another. I want you to act like a robot, interact with the scene and try your best to finish the tasks.\n"
        "The following instructions contains 4 parts. First part we'll tell you how we describe the scene, second part we'll tell you the task format, and in the third part we'll tell you how to interact with the scene, and at the last part we'll give you the scene's description.\n"
        "1. About Scene description\n"
        "We'll use a special structure, scene graph tree to describe the scene. In a scene graph tree = (V,E), vertices |V| are objects in the scene, directed edges |E| are the contact conditions of them, from the affording items to the objects on them.\n"
        "For example, if a room only contains a table with an apple on it, then the scene graph tree will contain 3 node. node 'Ground' will have an edge to node 'Table', and node 'Table' will have an edge to node 'Apple'. \n"
        "Following the descriptions, you will be given the information for every node, given in a bfs order.\n"
        "Ground node with(depth=0) is the root node, it will have some ground objects(depth=1) on it.\n"
        "For ground objects(depth=1), they may have a few affordable platforms and may or may not afford some non ground objects already.\n"
        "For non ground objects(depth>1), they will have a belonged ground object platform, and there will be information about the objects near it, divided into 8 directions.\n"
        "2. About Task format\n"
        "The task will be given in the following format. There are 5 types of tasks:\n"
        "0: Move object A to around object B. Tasks will be judged as success if you move A to the same platform with B. \n"
        "1: Move object A to B's {dir} space, where {dir} will be in 8 directions, namely['rear', 'rear-left', 'left', 'front-left', 'front', 'front-right', 'right', 'rear-right']. The task will be judge as success if you put A into the correct direction of B. We'll judge by where the last time you stand when putting down A.\n"
        "2: Move object A to an empty platform. Tasks will be judged as success if you put A to the correct platform.\n"
        "3: Move object A to an empty platform's {dir} space, where {dir} will be in 9 directions. The task will be judge as success if you put A into the correct direction of the platform. We'll judge by where the last time you stand when putting down A.\n"
        "4: Move object A between object B or C. Tasks will be judged as success if you move A between B or C. \n"
        "It is guaranteed that:\n"
        "For All type of tasks, the destination will be a platform on a ground object, and the item affords nothing, i.e. items are leaves in the scene graph tree.\n"
        "For task type 0, there will be at least one standing position for you to put down the item A around object B.\n"
        "For task type 1, 2, 3, there will be at least one standing position for you to put down the item A initially.\n"
        "For task type 4, there will be at least one position for you to put down the item A between object B or C initially\n"
        "3. About Interaction\n"
        "In the scene you will like running in an automaton. In each state, all the actions you can do can be described\n"
        "You'll be given different action space and corresponding images according to what stage you're on. Here are the following possible stages:\n"
        "Stage 'Going to the scene': In this stage, you'll need to decide where to go, or choose abandon and call end.\n"
        "Your action space will be [0, {li}] (where i is an integer). That is, you can choose to call end by '0', or choose to go to the ith platform by 'li'.\n"
        "Stage 'Item In wild': In this stage, you'll be given the view of the platform, standing in left/center/right of its freespace, marked or not marked the available objects, in total 6 pictures.\n"
        "You can choose one of the followings: pick up one item, go to other platforms, rotate your standing direction, call end.\n"
        "Your action space will be [-1, 0, {li}, {oi}]. That is, you can choose to call end by '0', choose to pick up choose to pick up the ith item by 'oi'.\n"
        "Note: \n"
        "So, if you think there's not enough space for you to put down the item, you can either try moving away the obstacles, or rotate your standing position, changing the 'front' means to you.\n"
        "1: You can pick up an item with other items on it, even though there will be no tasks force you to do so. "
        "However, the scene is static, so you'd better put it back to the same platform if you notice it, or you'll not be able to pick items that initially on it.\n"
        "2: The Platforms are ordered by height. e.g. A table have 3 platforms, then you'll see them as 'table_platform_0, table_platform_1, table_platform_2', among which 'table_platform_0' is the first platform from bottom to top.\n"
        "3: The system will let you try multiple times if you make a wrong choice, until you make a correct choice or reach a maximum interaction count.\n"
    )
    """
        bfs_queue = []
        bfs_queue.append(scene_graph_tree.nodes[root_node])
        bfs_queue_idx = 0

        while bfs_queue_idx < len(bfs_queue):
            current_node = bfs_queue[bfs_queue_idx]
            bfs_queue_idx += 1

            send_str_buffer += f"Node name: {current_node.name}\n"
            send_str_buffer += f"Node depth: {current_node.depth} (0 = ground, 1 = ground objects, >1 = non ground objects\n"
            send_str_buffer += f"Node parent: {current_node.parent.name if current_node.parent is not None else None}\n"
            send_str_buffer += (
                f"Node children: {[child.name for child in current_node.children]}\n"
            )
            send_str_buffer += f"Node platform number: {len(current_node.own_platform)}\n"

            platform_map = [[] for i in range(len(current_node.own_platform))]
            for child in current_node.children:
                belonged_platform = child.on_platform
                platform_map[current_node.own_platform.index(belonged_platform)].append(
                    child
                )

            for i in range(len(current_node.own_platform)):
                send_str_buffer += f"Platform{i} name: {current_node.own_platform[i].name}. It affords object: "
                for child in platform_map[i]:
                    send_str_buffer += f"{child.name}"
                    bfs_queue.append(child)
                send_str_buffer += "\n"
    """
    return send_str_buffer


# %%
# 0: parse replica
parse_replica.parse_replica(input_json_path, output_json_path)
# %%
# 1: load scene
scene = sapien.Scene()
scene.set_timestep(1 / 100)
scene.add_ground(altitude=0)

visualize_scene_sapien.load_objects_from_json(scene, output_json_path)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

for i in range(1000):
    scene.step()
    scene.update_render()
visualize_scene_sapien.reget_entities_from_sapien(
    scene, output_json_path, accurate_entity_path
)


# %%
# This is only for debug, reuse the atomic_task.pkl to avoid the scene_graph_generation.
scene_graph_tree = None

if os.path.exists("scene_graph_tree.pkl") and True:
    with open("scene_graph_tree.pkl", "rb") as f:
        scene_graph_tree = pickle.load(f)
else:
    # 3: reading glbs,  generate scene graph
    ts = time.perf_counter()
    json_tree_path = gen_scene_graph.load_json_file(accurate_entity_path)
    scene_graph_tree = gen_scene_graph.gen_multi_layer_graph_with_free_space(
        json_tree_path
    )
    glog.info(f"scene graph tree generation time:  {time.perf_counter() - ts}")

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
import ipdb

ipdb.set_trace()

# show_all(scene_graph_tree, save_path=f"{current_path}/scene_images/freespace_chk/")
# exit(0)


# %%
with open("scene_graph_tree.pkl", "wb") as f:
    scene_graph_tree.corresponding_scene = None
    pickle.dump(scene_graph_tree, f)

if os.path.exists("atomic_task.pkl") and True:
    with open("atomic_task.pkl", "rb") as f:
        atomic_task = pickle.load(f)
    #    tasks = atomic_task.generate_all_tasks()
else:
    atomic_task = atomic_task_generation.TaskGeneration(scene_graph_tree)
    atomic_task.generate_task_from_scene_graph_tree()


with open("atomic_task.pkl", "wb") as f:
    atomic_task.scene_graph_tree.corresponding_scene = None
    pickle.dump(atomic_task, f)


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

random.seed(125)
# 8: generate task sample and get sample images
# task_sample = random.sample(tasks, 2000)
# task_sample_ids = [tasks.index(task) for task in task_sample]
# print(task_sample_ids)

# The description is now stored at start_task_msg_buffer, which will be sent to VLM together with the first task's information.
task_description = give_vlm_scene_graph_tree_info()

import copy

start_task_msg_buffer = ""


scene_graph_tree.update_platform_children()

initial_atomic_task = copy.deepcopy(atomic_task)
initial_scene_graph_tree = copy.deepcopy(scene_graph_tree)
task_sample = random.sample(atomic_task.tasks, 200)
scene_graph_tree.corresponding_scene = scene
# import ipdb
# ipdb.set_trace()
for i in range(len(task_sample)):
    task = task_sample[i]
    # description of task has moved into apply function.

    global_vlm_interactor = vlm_interactor.VLMInteractor(mode="manual")
    global_vlm_interactor.send_only_message(task_description)

    glog.info(task)
    # return TaskStatusCode.SUCCESS or TaskStatusCode.FAILURE
    task = task_interact_helper.TaskInteractHelper(
        task=task,
        task_id=i,
        scene_graph_tree=scene_graph_tree,
        scene=scene,
        vlm_interactor=global_vlm_interactor,
        img_path=f"{current_path}",
    )
    task.apply_action(state=task_interact_helper.InteractStates.GOING_TO_THE_SCENE)

    scene = sapien.Scene()
    scene.set_timestep(1 / 100)
    scene.add_ground(altitude=0)

    visualize_scene_sapien.load_objects_from_json(scene, output_json_path)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    for j in range(1000):
        scene.step()
        scene.update_render()

    scene_graph_tree = copy.deepcopy(initial_scene_graph_tree)
    atomic_task = copy.deepcopy(initial_atomic_task)

    # import ipdb; ipdb.set_trace()
    # start_task_msg_buffer = ""
