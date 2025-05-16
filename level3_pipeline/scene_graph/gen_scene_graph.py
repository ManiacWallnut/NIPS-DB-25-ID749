# %%
"""
todo:
refactorize this.

implement check every object's


"""

# %%
import json
import math
import numpy as np

# from visualize_tree import visualize_tree
from .utils import visualize_tree
from scipy.spatial.transform import Rotation as R
import argparse
import copy
from .object import parse_objects, polygon_processor, affordable_platform
from .object.convex_hull_processor import ConvexHullProcessor_2d, Basic2DGeometry
from .object.meshprocessor import MeshProcessor
import sapien
import transforms3d

INF = 1e6
eps = 1e-6


# %%
# Function to load a JSON file from disk
def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


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
OPPOSITE_DIRECTIONS = {
    "front": "rear",
    "front-right": "rear-left",
    "right": "left",
    "rear-right": "front-left",
    "rear": "front",
    "rear-left": "front-right",
    "left": "right",
    "front-left": "rear-right",
}
FOUR_DIAGONAL_DIRECTIONS = ["rear-left", "front-left", "front-right", "rear-right"]


class TreeNode:
    def __init__(
        self,
        name,
        entity_config=None,
        removed=False,
        heading=None,
        free_space=None,
        parent=None,
        children=None,
        convex_hull_2d=None,
        bottom=None,
        top=None,
        bbox=None,
    ):
        self.name = name
        self.entity_config = entity_config
        self.object = parse_objects.SceneBox(
            name=name, heading=heading, convex_hull_2d=convex_hull_2d, bbox=bbox
        )
        self.heading = heading
        self.bottom = bottom
        self.top = top
        self.free_space = free_space
        self.free_space_height = [bottom, INF]
        self.critical_free_space = free_space

        if self.free_space is None and self.name != "GROUND":
            cos_theta, sin_theta = heading[0], heading[1]
            points = self.object.convex_hull_2d.get_headed_bbox_instance()

            def extend_point(p, direction):
                if direction == "right":
                    normal = np.array([cos_theta, sin_theta])
                elif direction == "left":
                    normal = np.array([-cos_theta, -sin_theta])
                elif direction == "front":
                    normal = np.array([-sin_theta, cos_theta])
                elif direction == "rear":
                    normal = np.array([sin_theta, -cos_theta])
                elif direction == "rear-left":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-left":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-right":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                elif direction == "rear-right":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                else:
                    return np.array(p)
                return np.array(p) + 10 * np.array(normal)

            # point older: rear_left, front_left, front_right, rear_right
            self.free_space = [
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "rear"),
                        points[0],
                        points[3],
                        extend_point(points[3], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(points[0], "rear"),
                        points[0],
                        points[3],
                        extend_point(points[3], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "left"),
                        extend_point(points[1], "left"),
                        points[1],
                        points[0],
                    ],
                    "Critical_space": [
                        extend_point(points[0], "left"),
                        extend_point(points[1], "left"),
                        points[1],
                        points[0],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                    "Critical_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[1],
                        extend_point(points[1], "front"),
                        extend_point(points[2], "front"),
                        points[2],
                    ],
                    "Critical_space": [
                        points[1],
                        extend_point(points[1], "front"),
                        extend_point(points[2], "front"),
                        points[2],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                    "Critical_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[3],
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[3], "right"),
                    ],
                    "Critical_space": [
                        points[3],
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[3], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[2], "rear"),
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[2], "rear-right"),
                    ],
                    "Critical_space": [
                        extend_point(points[2], "rear"),
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[2], "rear-right"),
                    ],
                },
            ]

        self.depth = 0
        self.parent = parent
        self.children = children if children is not None else []
        self.is_ambiguous = False
        self.on_platform = None
        self.own_platform = []
        self.bel_ground_object = None
        self.bel_ground_platform = None
        self.all_children = children if children is not None else []
        self.children_per_platform = []
        self.offspring_per_platform = []

    def get_bel_ground_object(self):
        if self.bel_ground_object is not None or self.depth <= 1:
            return self.bel_ground_object

        tmp_node = self.parent
        tmp_platform = self.on_platform
        while tmp_node.depth > 1:
            tmp_platform = tmp_node.on_platform
            tmp_node = tmp_node.parent

        self.bel_ground_platform = tmp_platform
        self.bel_ground_object = tmp_node
        return self.bel_ground_object

    def get_bel_ground_platform(self):
        if self.bel_ground_platform is not None or self.depth <= 1:
            return self.bel_ground_platform

        tmp_node = self.parent
        tmp_platform = self.on_platform
        while tmp_node.depth > 1:
            tmp_platform = tmp_node.on_platform
            tmp_node = tmp_node.parent

        self.bel_ground_platform = tmp_platform
        self.bel_ground_object = tmp_node
        return self.bel_ground_platform

    def update_children_belong_platform(self) -> None:
        self.children_per_platform = []
        for platform in self.own_platform:
            child_per_platform = []
            offspring_per_platform = []
            for child in self.children:
                if child.on_platform.name == platform.name:
                    child_per_platform.append(child)
                    child_queue = []
                    child_queue.append(child)
                    while len(child_queue) > 0:
                        cur_child = child_queue.pop(0)
                        offspring_per_platform.append(cur_child)
                        child_queue.extend(cur_child.children)
            self.children_per_platform.append(child_per_platform)
            self.offspring_per_platform.append(offspring_per_platform)
        return None

    def renew_heading(self, heading):
        #    for i in range(8):
        #        assert self.free_space[i]["Objects"] == []
        self.heading = heading
        self.object.convex_hull_2d.heading = heading
        if self.name != "GROUND":
            cos_theta, sin_theta = float(heading[0]), float(heading[1])
            points = self.object.convex_hull_2d.get_headed_bbox_instance()

            def extend_point(p, direction):
                if direction == "right":
                    normal = np.array([cos_theta, sin_theta])
                elif direction == "left":
                    normal = np.array([-cos_theta, -sin_theta])
                elif direction == "front":
                    normal = np.array([-sin_theta, cos_theta])
                elif direction == "rear":
                    normal = np.array([sin_theta, -cos_theta])
                elif direction == "rear-left":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-left":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [-cos_theta, -sin_theta]
                    )
                elif direction == "front-right":
                    normal = np.array([-sin_theta, cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                elif direction == "rear-right":
                    normal = np.array([sin_theta, -cos_theta]) + np.array(
                        [cos_theta, sin_theta]
                    )
                else:
                    return np.array(p)

                return np.array(p) + 10 * np.array(normal)

            # point older: rear_left, front_left, front_right, rear_right
            self.free_space = [
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "rear"),
                        points[0],
                        points[3],
                        extend_point(points[3], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(points[0], "rear"),
                        points[0],
                        points[3],
                        extend_point(points[3], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                    "Critical_space": [
                        extend_point(points[0], "rear-left"),
                        extend_point(points[0], "left"),
                        points[0],
                        extend_point(points[0], "rear"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[0], "left"),
                        extend_point(points[1], "left"),
                        points[1],
                        points[0],
                    ],
                    "Critical_space": [
                        extend_point(points[0], "left"),
                        extend_point(points[1], "left"),
                        points[1],
                        points[0],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                    "Critical_space": [
                        extend_point(points[1], "left"),
                        extend_point(points[1], "front-left"),
                        extend_point(points[1], "front"),
                        points[1],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[1],
                        extend_point(points[1], "front"),
                        extend_point(points[2], "front"),
                        points[2],
                    ],
                    "Critical_space": [
                        points[1],
                        extend_point(points[1], "front"),
                        extend_point(points[2], "front"),
                        points[2],
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                    "Critical_space": [
                        points[2],
                        extend_point(points[2], "front"),
                        extend_point(points[2], "front-right"),
                        extend_point(points[2], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        points[3],
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[3], "right"),
                    ],
                    "Critical_space": [
                        points[3],
                        points[2],
                        extend_point(points[2], "right"),
                        extend_point(points[3], "right"),
                    ],
                },
                {
                    "Objects": [],
                    "Available_space": [
                        extend_point(points[3], "rear"),
                        points[3],
                        extend_point(points[3], "right"),
                        extend_point(points[3], "rear-right"),
                    ],
                    "Critical_space": [
                        extend_point(points[3], "rear"),
                        points[3],
                        extend_point(points[3], "right"),
                        extend_point(points[3], "rear-right"),
                    ],
                },
            ]

    def reset_free_space(self):
        cos_theta, sin_theta = self.heading[0], self.heading[1]
        points = self.object.convex_hull_2d.get_headed_bbox_instance()

        def extend_point(p, direction):
            if direction == "right":
                normal = np.array([cos_theta, sin_theta])
            elif direction == "left":
                normal = np.array([-cos_theta, -sin_theta])
            elif direction == "front":
                normal = np.array([-sin_theta, cos_theta])
            elif direction == "rear":
                normal = np.array([sin_theta, -cos_theta])
            elif direction == "rear-left":
                normal = np.array([sin_theta, -cos_theta]) + np.array(
                    [-cos_theta, -sin_theta]
                )
            elif direction == "front-left":
                normal = np.array([-sin_theta, cos_theta]) + np.array(
                    [-cos_theta, -sin_theta]
                )
            elif direction == "front-right":
                normal = np.array([-sin_theta, cos_theta]) + np.array(
                    [cos_theta, sin_theta]
                )
            elif direction == "rear-right":
                normal = np.array([sin_theta, -cos_theta]) + np.array(
                    [cos_theta, sin_theta]
                )
            else:
                return np.array(p)
            return np.array(p) + 10 * np.array(normal)

        # point older: rear_left, front_left, front_right, rear_right
        self.free_space = [
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[0], "rear"),
                    points[0],
                    points[3],
                    extend_point(points[3], "rear"),
                ],
                "Critical_space": [
                    extend_point(points[0], "rear"),
                    points[0],
                    points[3],
                    extend_point(points[3], "rear"),
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[0], "rear-left"),
                    extend_point(points[0], "left"),
                    points[0],
                    extend_point(points[0], "rear"),
                ],
                "Critical_space": [
                    extend_point(points[0], "rear-left"),
                    extend_point(points[0], "left"),
                    points[0],
                    extend_point(points[0], "rear"),
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[0], "left"),
                    extend_point(points[1], "left"),
                    points[1],
                    points[0],
                ],
                "Critical_space": [
                    extend_point(points[0], "left"),
                    extend_point(points[1], "left"),
                    points[1],
                    points[0],
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[1], "left"),
                    extend_point(points[1], "front-left"),
                    extend_point(points[1], "front"),
                    points[1],
                ],
                "Critical_space": [
                    extend_point(points[1], "left"),
                    extend_point(points[1], "front-left"),
                    extend_point(points[1], "front"),
                    points[1],
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    points[1],
                    extend_point(points[1], "front"),
                    extend_point(points[2], "front"),
                    points[2],
                ],
                "Critical_space": [
                    points[1],
                    extend_point(points[1], "front"),
                    extend_point(points[2], "front"),
                    points[2],
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    points[2],
                    extend_point(points[2], "front"),
                    extend_point(points[2], "front-right"),
                    extend_point(points[2], "right"),
                ],
                "Critical_space": [
                    points[2],
                    extend_point(points[2], "front"),
                    extend_point(points[2], "front-right"),
                    extend_point(points[2], "right"),
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    points[3],
                    points[2],
                    extend_point(points[2], "right"),
                    extend_point(points[3], "right"),
                ],
                "Critical_space": [
                    points[3],
                    points[2],
                    extend_point(points[2], "right"),
                    extend_point(points[3], "right"),
                ],
            },
            {
                "Objects": [],
                "Available_space": [
                    extend_point(points[3], "rear"),
                    points[3],
                    extend_point(points[3], "right"),
                    extend_point(points[3], "rear-right"),
                ],
                "Critical_space": [
                    extend_point(points[3], "rear"),
                    points[3],
                    extend_point(points[3], "right"),
                    extend_point(points[3], "rear-right"),
                ],
            },
        ]

    def update_free_space(self, other_node_name, direction):
        assert (
            len(self.free_space[direction]["Available_space"]) == 4
            and direction >= 0
            and direction < 8
        )
        # print(self.name, direction, other_node.name, self.free_space[EIGHT_DIRECTIONS.index(direction)]["Available_space"])
        self.free_space[direction]["Objects"].append(other_node_name)

    def get_critical_space_len(self, direction):
        return [
            np.linalg.norm(
                self.free_space[direction]["Critical_space"][0]
                - self.free_space[direction]["Critical_space"][1]
            ),
            np.linalg.norm(
                self.free_space[direction]["Critical_space"][1]
                - self.free_space[direction]["Critical_space"][2]
            ),
        ]

    def get_standing_point(self, direction):
        if direction == 0:
            return (
                self.free_space[0]["Critical_space"][1]
                + self.free_space[0]["Critical_space"][2]
            ) / 2 + (
                self.free_space[0]["Critical_space"][0]
                - self.free_space[0]["Critical_space"][1]
            ) * 0.35 / np.linalg.norm(
                self.free_space[0]["Critical_space"][0]
                - self.free_space[0]["Critical_space"][1]
            )
        elif direction == 2:
            return (
                self.free_space[2]["Critical_space"][2]
                + self.free_space[2]["Critical_space"][3]
            ) / 2 + (
                self.free_space[2]["Critical_space"][1]
                - self.free_space[2]["Critical_space"][2]
            ) * 0.35 / np.linalg.norm(
                self.free_space[2]["Critical_space"][1]
                - self.free_space[2]["Critical_space"][2]
            )
        elif direction == 4:
            return (
                self.free_space[4]["Critical_space"][3]
                + self.free_space[4]["Critical_space"][0]
            ) / 2 + (
                self.free_space[4]["Critical_space"][2]
                - self.free_space[4]["Critical_space"][3]
            ) * 0.35 / np.linalg.norm(
                self.free_space[4]["Critical_space"][2]
                - self.free_space[4]["Critical_space"][3]
            )
        elif direction == 6:
            return (
                self.free_space[6]["Critical_space"][0]
                + self.free_space[6]["Critical_space"][1]
            ) / 2 + (
                self.free_space[6]["Critical_space"][3]
                - self.free_space[6]["Critical_space"][0]
            ) * 0.35 / np.linalg.norm(
                self.free_space[6]["Critical_space"][3]
                - self.free_space[6]["Critical_space"][0]
            )
        else:
            return None

    def get_center(self):
        return (
            self.free_space[0]["Critical_space"][1]
            + self.free_space[4]["Critical_space"][3]
        ) / 2

    def get_critical_space_center(self, direction):
        assert (
            len(self.free_space[direction]["Critical_space"]) == 4
            and direction >= 0
            and direction < 8
        )
        return np.mean(self.free_space[direction]["Critical_space"], axis=0)

    def get_critical_space_left(self, direction):
        assert (
            len(self.free_space[direction]["Critical_space"]) == 4
            and direction >= 0
            and direction < 8
        )
        return (
            np.mean(self.free_space[direction]["Critical_space"][:2], axis=0)
            + np.mean(self.free_space[direction]["Critical_space"], axis=0)
        ) / 2

    def get_critical_space_right(self, direction):
        assert (
            len(self.free_space[direction]["Critical_space"]) == 4
            and direction >= 0
            and direction < 8
        )
        return (
            np.mean(self.free_space[direction]["Critical_space"][2:], axis=0)
            + np.mean(self.free_space[direction]["Critical_space"], axis=0)
        ) / 2

    def freespace_is_standable(self, direction):
        assert (
            len(self.free_space[direction]["Critical_space"]) == 4
            and direction >= 0
            and direction < 8
        )
        if direction == 0 or direction == 4:
            return (
                np.linalg.norm(
                    self.free_space[direction]["Critical_space"][0]
                    - self.free_space[direction]["Critical_space"][1]
                )
                > 0.35
            )
        elif direction == 2 or direction == 6:
            return (
                np.linalg.norm(
                    self.free_space[direction]["Critical_space"][1]
                    - self.free_space[direction]["Critical_space"][2]
                )
                > 0.35
            )
        else:
            return 0

    def get_critical_space_bbox(self):
        return [
            self.free_space[1]["Critical_space"][0],
            self.free_space[3]["Critical_space"][1],
            self.free_space[5]["Critical_space"][2],
            self.free_space[7]["Critical_space"][3],
        ]

    def sweep_platform(self):
        max_size = -INF
        for platform in self.own_platform:
            max_size = max(
                max_size,
                (platform.bbox[1][0] - platform.bbox[0][0])
                * (platform.bbox[1][1] - platform.bbox[0][1]),
            )
        self.own_platform = [
            platform
            for platform in self.own_platform
            if (platform.bbox[1][0] - platform.bbox[0][0])
            * (platform.bbox[1][1] - platform.bbox[0][1])
            > max_size * 0.25
        ]

    def at_which_part(self):
        platform_bbox = self.on_platform.convex_hull_2d.get_headed_bbox_instance()
        center = self.get_center()
        direction_mappings = {
            (0, 0): "rear-left",
            (0, 1): "rear",
            (0, 2): "rear-right",
            (1, 0): "left",
            (1, 1): "center",
            (1, 2): "right",
            (2, 0): "front-left",
            (2, 1): "front",
            (2, 2): "front-right",
        }
        for i in range(3):
            for j in range(3):
                rect = [
                    platform_bbox[0]
                    + i * (platform_bbox[1] - platform_bbox[0]) / 3
                    + j * (platform_bbox[3] - platform_bbox[0]) / 3,
                    platform_bbox[0]
                    + (i + 1) * (platform_bbox[1] - platform_bbox[0]) / 3
                    + j * (platform_bbox[3] - platform_bbox[0]) / 3,
                    platform_bbox[0]
                    + (i + 1) * (platform_bbox[1] - platform_bbox[0]) / 3
                    + (j + 1) * (platform_bbox[3] - platform_bbox[0]) / 3,
                    platform_bbox[0]
                    + i * (platform_bbox[1] - platform_bbox[0]) / 3
                    + (j + 1) * (platform_bbox[3] - platform_bbox[0]) / 3,
                ]
                if Basic2DGeometry.is_inside_rectangle(center, rect):
                    return direction_mappings[(i, j)]
        print("warning: at_which_part failed")
        return "center"

    def get_fitable_pose(self, avl_height, place_bbox):
        if self.top - self.bottom > avl_height:
            return None
        return self.object.convex_hull_2d.get_fit_in_translation(place_bbox)

    def rotate_free_space(self, front):
        if front == 0:
            return
        elif front == 4:
            self.heading = [-self.heading[0], -self.heading[1]]
            self.free_space = self.free_space[4:] + self.free_space[:4]
            for i in range(8):
                self.free_space[i]["Available_space"] = (
                    self.free_space[i]["Available_space"][2:]
                    + self.free_space[i]["Available_space"][:2]
                )
                self.free_space[i]["Critical_space"] = (
                    self.free_space[i]["Critical_space"][2:]
                    + self.free_space[i]["Critical_space"][:2]
                )
        elif front == 2:
            self.heading = [self.heading[1], -self.heading[0]]
            self.free_space = self.free_space[2:] + self.free_space[:2]
            for i in range(8):
                self.free_space[i]["Available_space"] = (
                    self.free_space[i]["Available_space"][1:]
                    + self.free_space[i]["Available_space"][:1]
                )
                self.free_space[i]["Critical_space"] = (
                    self.free_space[i]["Critical_space"][1:]
                    + self.free_space[i]["Critical_space"][:1]
                )
        elif front == 6:
            self.heading = [-self.heading[1], self.heading[0]]
            self.free_space = self.free_space[6:] + self.free_space[:6]
            for i in range(8):
                self.free_space[i]["Available_space"] = (
                    self.free_space[i]["Available_space"][3:]
                    + self.free_space[i]["Available_space"][:3]
                )
                self.free_space[i]["Critical_space"] = (
                    self.free_space[i]["Critical_space"][3:]
                    + self.free_space[i]["Critical_space"][:3]
                )

    def rotate_free_space_for_all_children(self, front):
        self.rotate_free_space(front)
        for child in self.children:
            child.rotate_free_space_for_all_children(front)

    def set_free_space_to_platform(self, platform_hull):
        platform_hull.heading = self.heading
        platform_bbox = platform_hull.get_headed_bbox_instance()

        # rear, rearleft, left, frontleft, front, frontright, right, rearright
        # left=123, front=345, right=567, rear=701
        platform_left_side = [platform_bbox[0], platform_bbox[1]]
        platform_right_side = [platform_bbox[2], platform_bbox[3]]
        platform_front_side = [platform_bbox[1], platform_bbox[2]]
        platform_rear_side = [platform_bbox[3], platform_bbox[0]]
        # if self.name == 'frl_apartment_choppingboard_02_81':
        #     print(self.name, self.heading, 'before\n',self.free_space)
        #     print(platform_left_side, platform_right_side)
        #     print(platform_front_side, platform_rear_side)
        for left_part in range(1, 4):

            space_front_side = [
                self.free_space[left_part]["Available_space"][1],
                self.free_space[left_part]["Available_space"][2],
            ]
            space_rear_side = [
                self.free_space[left_part]["Available_space"][0],
                self.free_space[left_part]["Available_space"][3],
            ]

            intersect_left_front = Basic2DGeometry.intersection_of_line(
                platform_left_side, space_front_side
            )
            intersect_left_rear = Basic2DGeometry.intersection_of_line(
                platform_left_side, space_rear_side
            )
            #    if self.name == 'frl_apartment_choppingboard_02_81':
            #        print(intersect_left_front, platform_left_side, space_front_side)
            if intersect_left_front is not None and intersect_left_rear is not None:
                if Basic2DGeometry.is_on_segment(
                    intersect_left_front, space_front_side
                ) and Basic2DGeometry.is_on_segment(
                    intersect_left_rear, space_rear_side
                ):
                    (
                        self.free_space[left_part]["Available_space"][0],
                        self.free_space[left_part]["Available_space"][1],
                    ) = (intersect_left_rear, intersect_left_front)
                elif Basic2DGeometry.is_on_segment(
                    space_front_side[1], [space_front_side[0], intersect_left_front]
                ):
                    (
                        self.free_space[left_part]["Available_space"][0],
                        self.free_space[left_part]["Available_space"][1],
                    ) = (
                        self.free_space[left_part]["Available_space"][3],
                        self.free_space[left_part]["Available_space"][2],
                    )

        for right_part in range(5, 8):

            space_front_side = [
                self.free_space[right_part]["Available_space"][1],
                self.free_space[right_part]["Available_space"][2],
            ]
            space_rear_side = [
                self.free_space[right_part]["Available_space"][0],
                self.free_space[right_part]["Available_space"][3],
            ]
            intersect_right_front = Basic2DGeometry.intersection_of_line(
                platform_right_side, space_front_side
            )
            intersect_right_rear = Basic2DGeometry.intersection_of_line(
                platform_right_side, space_rear_side
            )
            if intersect_left_front is not None and intersect_left_rear is not None:
                if Basic2DGeometry.is_on_segment(
                    intersect_right_front, space_front_side
                ) and Basic2DGeometry.is_on_segment(
                    intersect_right_rear, space_rear_side
                ):
                    (
                        self.free_space[right_part]["Available_space"][3],
                        self.free_space[right_part]["Available_space"][2],
                    ) = (intersect_right_rear, intersect_right_front)
                elif Basic2DGeometry.is_on_segment(
                    space_front_side[0], [space_front_side[1], intersect_right_front]
                ):
                    (
                        self.free_space[right_part]["Available_space"][2],
                        self.free_space[right_part]["Available_space"][3],
                    ) = (
                        self.free_space[right_part]["Available_space"][1],
                        self.free_space[right_part]["Available_space"][0],
                    )

        for front_part in range(3, 6):

            space_left_side = [
                self.free_space[front_part]["Available_space"][0],
                self.free_space[front_part]["Available_space"][1],
            ]
            space_right_side = [
                self.free_space[front_part]["Available_space"][3],
                self.free_space[front_part]["Available_space"][2],
            ]
            intersect_front_left = Basic2DGeometry.intersection_of_line(
                platform_front_side, space_left_side
            )
            intersect_front_right = Basic2DGeometry.intersection_of_line(
                platform_front_side, space_right_side
            )
            if intersect_front_left is not None and intersect_front_right is not None:
                if Basic2DGeometry.is_on_segment(
                    intersect_front_left, space_left_side
                ) and Basic2DGeometry.is_on_segment(
                    intersect_front_right, space_right_side
                ):
                    (
                        self.free_space[front_part]["Available_space"][1],
                        self.free_space[front_part]["Available_space"][2],
                    ) = (intersect_front_left, intersect_front_right)
                elif Basic2DGeometry.is_on_segment(
                    space_left_side[0], [space_left_side[1], intersect_front_left]
                ):
                    (
                        self.free_space[front_part]["Available_space"][1],
                        self.free_space[front_part]["Available_space"][2],
                    ) = (
                        self.free_space[front_part]["Available_space"][0],
                        self.free_space[front_part]["Available_space"][3],
                    )

        for rear_part in [7, 0, 1]:

            space_left_side = [
                self.free_space[rear_part]["Available_space"][0],
                self.free_space[rear_part]["Available_space"][1],
            ]
            space_right_side = [
                self.free_space[rear_part]["Available_space"][3],
                self.free_space[rear_part]["Available_space"][2],
            ]
            intersect_rear_left = Basic2DGeometry.intersection_of_line(
                platform_rear_side, space_left_side
            )
            intersect_rear_right = Basic2DGeometry.intersection_of_line(
                platform_rear_side, space_right_side
            )
            if intersect_rear_left is not None and intersect_rear_right is not None:
                if Basic2DGeometry.is_on_segment(
                    intersect_rear_left, space_left_side
                ) and Basic2DGeometry.is_on_segment(
                    intersect_rear_right, space_right_side
                ):
                    (
                        self.free_space[rear_part]["Available_space"][0],
                        self.free_space[rear_part]["Available_space"][3],
                    ) = (intersect_rear_left, intersect_rear_right)
                elif Basic2DGeometry.is_on_segment(
                    space_left_side[1], [space_left_side[0], intersect_rear_left]
                ):
                    (
                        self.free_space[rear_part]["Available_space"][0],
                        self.free_space[rear_part]["Available_space"][3],
                    ) = (
                        self.free_space[rear_part]["Available_space"][1],
                        self.free_space[rear_part]["Available_space"][2],
                    )

        pass

    def sync_critical_free_space(self):
        for dir in range(8):
            self.free_space[dir]["Critical_space"] = copy.deepcopy(
                self.free_space[dir]["Available_space"]
            )

    def clean_free_space(self):
        if self.free_space == None:
            return

        def rotate_point_clockwise(point, theta):
            rotation_matrix = np.array(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )
            return np.dot(rotation_matrix, point)

        def has_area(points, heading):
            left_rear, left_front, right_front, right_rear = points
            cos_theta, sin_theta = heading[0], heading[1]

            # 计算旋转角度 theta
            theta = np.arctan2(sin_theta, cos_theta)

            # 将矩形的四个角的坐标按照 theta 顺时针旋转到标准位置
            left_rear_rotated = rotate_point_clockwise(left_rear, theta)
            left_front_rotated = rotate_point_clockwise(left_front, theta)
            right_rear_rotated = rotate_point_clockwise(right_rear, theta)
            right_front_rotated = rotate_point_clockwise(right_front, theta)

            # 检查右后和右前点的线段是否在左后和左前段的右侧
            if (
                right_rear_rotated[0] < left_rear_rotated[0]
                or right_front_rotated[0] < left_front_rotated[0]
            ):
                return False
            if (
                right_rear_rotated[1] > left_rear_rotated[1]
                or right_front_rotated[1] < left_front_rotated[1]
            ):
                return False

            return True

        for direction in range(len(self.free_space)):
            if self.free_space[direction]["Available_space"] != "not available":
                #   print(self.name, direction, self.free_space[direction]['Available_space'])
                if not has_area(
                    self.free_space[direction]["Available_space"], self.heading
                ):
                    self.free_space[direction]["Objects"] = []
                    self.free_space[direction]["Available_space"] = "not available"
            if self.free_space[direction]["Critical_space"] != "not available":
                if not has_area(
                    self.free_space[direction]["Critical_space"], self.heading
                ):
                    self.free_space[direction]["Objects"] = []
                    self.free_space[direction]["Critical_space"] = "not available"

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def get_child(self, child_name):
        for child in self.children:
            if child.name == child_name:
                return child

    def get_available_place_for_object_on_platform(self, object_node, platform_id):
        self.update_children_belong_platform()
        platform_children = self.own_platform[platform_id]
        obstacle_obj_list = []
        platform_result = []
        for child in self.children_per_platform[platform_id]:
            if child.name != object_node.name:
                obstacle_obj_list.append(child)
        platform_result = platform_children.get_all_available_freespace_combinations(
            obstacle_obj_list, object_node.object, min_step=0.01
        )
        return platform_result

    def get_available_place_for_object(self, object_node):
        self.update_children_belong_platform()
        result = []
        for i, platform_children in enumerate(self.own_platform):
            platform_result = []
            obstacle_obj_list = []
            for child in self.children_per_platform[i]:
                if child.name != object_node.name:
                    obstacle_obj_list.append(child)
            platform_result = (
                platform_children.get_all_available_freespace_combinations(
                    obstacle_obj_list, object_node.object, min_step=0.01
                )
            )
            result.append(platform_result)

        print(result)
        return result

    #            if len(obstacle_bbox_list) == 0:
    #               result = []

    def remove_child(self, child_node):
        for i in range(len(self.children)):
            if self.children[i].name == child_node.name:
                self.children[i].parent = None
                self.children.pop(i)
                break

    def display_unique_information(self, ambiguous=False):
        if self.depth == 0:
            return ""
        else:
            str_buffer = ""
            str_buffer += f"object: {self.name}\n"
            str_buffer += f"belonged_to_platform: {self.on_platform.name if self.on_platform is not None and self.depth > 1 else None}\n"
            if self.is_ambiguous:
                str_buffer += f"Its surroundings\n"
                for dir in range(8):
                    if (
                        self.free_space[dir]["Objects"] is not None
                        and len(self.free_space[dir]["Objects"]) > 0
                    ):
                        str_buffer += f'{EIGHT_DIRECTIONS[dir]}, {[objs.name for objs in self.free_space[dir]["Objects"]]}\n'
            if self.depth > 1:
                str_buffer += f"Its direct parent:\n"
            str_buffer += self.parent.display_unique_information(ambiguous) + "\n"

        return str_buffer


class Tree:

    def __init__(self):
        self.nodes = {}  # Dictionary to access nodes by name
        self.edges = {}
        self.corresponding_scene = None

    def load_corresponding_scene(self, scene):
        self.corresponding_scene = scene
        # Dictionary to access edges by name tuple(name1, name2, platform), edge[(name1, name2)]=platform means name1 is on a platform named platform, and platform is in name2

        # (self, name, heading=None, free_space=None, parent=None, children=None,convex_hull_2d=None, top = None, bbox=None):

    def from_scene_platform_list(self, object_platform_list, contacts):
        for item in object_platform_list:
            if item.bel_object == None:
                item_node = TreeNode(
                    name=item.name,
                    heading=item.heading,
                    free_space=None,
                    bbox=item.bbox,
                    convex_hull_2d=item.convex_hull_2d,
                    bottom=item.height,
                    top=item.top,
                )
                self.set_node(item_node)
            else:
                self.nodes[item.bel_object].own_platform.append(item)

        # sort platforms by height and rename them.
        for node in self.nodes:
            platforms = self.nodes[node].own_platform
            platforms = parse_objects.ScenePlatform.sort_platforms(platforms)
            for i in range(len(platforms)):
                platforms[i].name = node + "_platform_" + str(i)

            self.nodes[node].own_platform = platforms

        for contact in contacts:
            if contact[0].bel_object is None:
                print("contact[0] should be a platform but belong is None")
                continue
            if contact[1].bel_object is not None:
                continue
                print("contact[1] should be a object but belong is not None")
            if (
                self.nodes[self.nodes[contact[0].bel_object].name].parent
                == self.nodes[contact[1].name]
            ):
                continue
            self.edges[(self.nodes[contact[0].bel_object].name, contact[1].name)] = (
                contact[0].name
            )
            self.nodes[self.nodes[contact[0].bel_object].name].add_child(
                self.nodes[contact[1].name]
            )
            self.nodes[contact[1].name].parent = self.nodes[
                self.nodes[contact[0].bel_object].name
            ]
            self.nodes[contact[1].name].on_platform = contact[0]
            self.nodes[contact[1].name].free_space_height = [
                contact[0].height,
                contact[0].height + contact[0].avl_height,
            ]

        on_platform_dict = {}
        ambiguous_items = set()
        for node in self.nodes.values():
            if node.on_platform is not None:
                pure_name = node.name[: node.name.rfind("_")]
                if pure_name not in on_platform_dict:
                    on_platform_dict[pure_name] = [node.on_platform]
                else:
                    for platform_have_this_item in on_platform_dict[pure_name]:
                        if platform_have_this_item.name == node.on_platform.name:
                            ambiguous_items.add((pure_name, node.on_platform.name))
                    on_platform_dict[pure_name].append(node.on_platform)

        print("ambiguous items:", ambiguous_items)

        for node in self.nodes.values():
            pure_name = node.name[: node.name.rfind("_")]
            if (
                node.on_platform is not None
                and (pure_name, node.on_platform.name) in ambiguous_items
            ):
                node.is_ambiguous = True
            else:
                node.is_ambiguous = False

        pass

    def dfs_for_freespace(self, node):
        print(node.name)
        for child in node.children:
            child.depth = node.depth + 1
            if node.name == "GROUND":
                self.dfs_for_freespace(child)
            else:
                child.renew_heading(node.heading)

                self.dfs_for_freespace(child)
        if node.depth >= 1:
            for i in range(len(node.children)):
                child = node.children[i]
                #   if 'picture_03' in child.name:
                #       print([child.free_space[j]['Available_space'] for j in range(8)])
                child.reset_free_space()
                #   if 'picture_03' in child.name:
                #       print([child.free_space[j]['Available_space'] for j in range(8)])
                # if 'picture' in child.name:
                #     print([child.free_space[j]['Available_space'] for j in range(8)])
                #  print(node.name, node.heading, node.object.heading)
                child.set_free_space_to_platform(child.on_platform.convex_hull_2d)
                #   if 'picture_03' in child.name:
                #       print(child.on_platform.name)
                #       print(child.on_platform.convex_hull_2d.get_headed_bbox_instance())
                #       print([child.free_space[j]['Available_space'] for j in range(8)])
                child.sync_critical_free_space()

            #   if 'picture_03' in child.name:
            #       print([child.free_space[j]['Available_space'] for j in range(8)])

        for i in range(len(node.children)):
            for j in range(len(node.children)):
                if i == j:
                    continue
                child1, child2 = node.children[i], node.children[j]
                child1_obj, child2_obj = child1.object, child2.object

                # Must belong to the same platform.
                if child1.on_platform != child2.on_platform:
                    continue

                surface_directions = []
                for k in range(8):
                    rect = child1.free_space[k]["Available_space"]
                    #    if 'picture_03' in child1.name:
                    #        print(child1.name, child2.name, rect, child2_obj.convex_hull_2d.get_headed_bbox_instance())
                    #        print(child2_obj.convex_hull_2d.is_intersected_with_rectangle(rect))
                    if child2_obj.convex_hull_2d.is_intersected_with_rectangle(rect):
                        surface_directions.append(k)

                for k in range(0, 8, 2):
                    if (
                        k not in surface_directions
                        and (k + 1) % 8 in surface_directions
                        and (k + 7) % 8 in surface_directions
                    ):
                        surface_directions.append(k)

                # if child1.name == 'frl_apartment_wall_cabinet_01_5':
                #    print(child2.name, surface_directions)

                for direction in surface_directions:
                    #          print(child1.name, child2.name, direction)
                    child1.update_free_space(child2, direction)

        #   if 'picture_03' in node.children[i].name:
        #       print(node.children[i].name)
        #       print([node.children[i].free_space[j]['Objects'] for j in range(8)])

        # cut free space among children

        # bbox_min, bbox_max = node.object.bbox

        # cut
        # rear-left = 0, front-left = 1, front-right = 2, rear-right = 3

        for i in range(len(node.children)):
            child = node.children[i]
            for left_part in range(1, 4):

                for object_node in child.free_space[left_part]["Objects"]:
                    near_side = [
                        child.free_space[left_part]["Critical_space"][3],
                        child.free_space[left_part]["Critical_space"][2],
                    ]
                    far_side = [
                        child.free_space[left_part]["Critical_space"][0],
                        child.free_space[left_part]["Critical_space"][1],
                    ]

                    # if child.name == 'frl_apartment_wall_cabinet_01_5':
                    #    print(object_node.name, EIGHT_DIRECTIONS[left_part])
                    #    print('near',near_side,'far', far_side)
                    #    print('convexhull', object_node.object.convex_hull_2d.get_vertices_on_convex_hull())
                    new_near_side, new_far_side = (
                        object_node.object.convex_hull_2d.cut_free_space_with_convex(
                            near_side, far_side, force=left_part % 2 == 0
                        )
                    )
                    child.free_space[left_part]["Critical_space"] = [
                        new_far_side[0],
                        new_far_side[1],
                        new_near_side[1],
                        new_near_side[0],
                    ]
                    # if child.name == 'frl_apartment_wall_cabinet_01_5':
                    #    print('new near far', new_near_side, new_far_side)

            for right_part in range(5, 8):

                for object_node in child.free_space[right_part]["Objects"]:
                    near_side = [
                        child.free_space[right_part]["Critical_space"][0],
                        child.free_space[right_part]["Critical_space"][1],
                    ]
                    far_side = [
                        child.free_space[right_part]["Critical_space"][3],
                        child.free_space[right_part]["Critical_space"][2],
                    ]

                    # if child.name == 'frl_apartment_wall_cabinet_01_5':
                    #    print(object_node.name, EIGHT_DIRECTIONS[right_part])
                    #    print(near_side, far_side)
                    #    print(object_node.object.convex_hull_2d.get_vertices_on_convex_hull())
                    new_near_side, new_far_side = (
                        object_node.object.convex_hull_2d.cut_free_space_with_convex(
                            near_side, far_side, force=right_part % 2 == 0
                        )
                    )
                    child.free_space[right_part]["Critical_space"] = [
                        new_near_side[0],
                        new_near_side[1],
                        new_far_side[1],
                        new_far_side[0],
                    ]
                    # if child.name == 'frl_apartment_wall_cabinet_01_5':
                    #    print('new near far', new_near_side, new_far_side)
            for front_part in range(3, 6):

                for object_node in child.free_space[front_part]["Objects"]:
                    near_side = [
                        child.free_space[front_part]["Critical_space"][0],
                        child.free_space[front_part]["Critical_space"][3],
                    ]
                    far_side = [
                        child.free_space[front_part]["Critical_space"][1],
                        child.free_space[front_part]["Critical_space"][2],
                    ]

                    #   if child.name == 'frl_apartment_wall_cabinet_01_5':
                    #       print(object_node.name, EIGHT_DIRECTIONS[front_part])
                    #      print(near_side, far_side)
                    #      print(object_node.object.convex_hull_2d.get_vertices_on_convex_hull())
                    new_near_side, new_far_side = (
                        object_node.object.convex_hull_2d.cut_free_space_with_convex(
                            near_side, far_side, force=front_part % 2 == 0
                        )
                    )
                    child.free_space[front_part]["Critical_space"] = [
                        new_near_side[0],
                        new_far_side[0],
                        new_far_side[1],
                        new_near_side[1],
                    ]

            for rear_part in [7, 0, 1]:

                for object_node in child.free_space[rear_part]["Objects"]:
                    near_side = [
                        child.free_space[rear_part]["Critical_space"][1],
                        child.free_space[rear_part]["Critical_space"][2],
                    ]
                    far_side = [
                        child.free_space[rear_part]["Critical_space"][0],
                        child.free_space[rear_part]["Critical_space"][3],
                    ]

                    # if child.name == 'frl_apartment_wall_cabinet_01_5' and rear_part == 0:
                    #     print(object_node.name, 'rear')
                    #    print(near_side, far_side)
                    #    print(object_node.object.convex_hull_2d.get_vertices_on_convex_hull())
                    new_near_side, new_far_side = (
                        object_node.object.convex_hull_2d.cut_free_space_with_convex(
                            near_side, far_side, force=rear_part % 2 == 0
                        )
                    )

                    child.free_space[rear_part]["Critical_space"] = [
                        new_far_side[0],
                        new_near_side[0],
                        new_near_side[1],
                        new_far_side[1],
                    ]

        return None

    def calculate_free_space(self):
        for node in self.nodes.values():
            # print(node.name, node.parent.name if node.parent is not None else None, node.on_platform)
            if node.parent is None:
                self.dfs_for_freespace(node)
        return None

    def clean_zero_area_free_space(self):
        for node in self.nodes.values():
            node.clean_free_space()

    def cut_free_space_with_stage(self, stage):

        min_intersection_height = INF
        max_stage_height = -INF
        for stage_obj in stage:
            for geometry in stage_obj.mesh:
                max_stage_height = max(max_stage_height, geometry.mesh.bounds[1][2])
                for node_name, node in self.nodes.items():
                    if node.free_space is None:
                        continue
                    bottom, top = node.bottom, node.top
                    if bottom > top:
                        continue

                    node_bbox = node.object.convex_hull_2d.get_headed_bbox_instance()

                    node_bbox_8_directions = [
                        node_bbox[0],
                        (node_bbox[0] + node_bbox[1]) * 0.5,
                        node_bbox[1],
                        (node_bbox[1] + node_bbox[2]) * 0.5,
                        node_bbox[2],
                        (node_bbox[2] + node_bbox[3]) * 0.5,
                        node_bbox[3],
                        (node_bbox[3] + node_bbox[0]) * 0.5,
                    ]

                    for left_part in range(1, 4):
                        rect = node.free_space[left_part]["Critical_space"]
                        if isinstance(rect, str):
                            continue

                        cuboid_vertices = [
                            [rect[0][0], rect[0][1], bottom],
                            [rect[1][0], rect[1][1], bottom],
                            [rect[2][0], rect[2][1], bottom],
                            [rect[3][0], rect[3][1], bottom],
                            [rect[0][0], rect[0][1], top],
                            [rect[1][0], rect[1][1], top],
                            [rect[2][0], rect[2][1], top],
                            [rect[3][0], rect[3][1], top],
                        ]
                        # cuboid_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]

                        if (
                            np.min(np.array(rect)) < -1e10
                            or np.max(np.array(rect)) > 1e10
                            or bottom < -1e10
                            or top > 1e10
                        ):
                            print("overflow", node_name)

                        cuboid = MeshProcessor.create_cuboid_from_vertices(
                            cuboid_vertices
                        )
                        # print(bottom, top, rect,    cuboid.mesh.volume)
                        intersect = geometry.intersection_with_cuboid(cuboid)

                        if (
                            intersect is not None
                            and intersect.mesh.bounds[1][2]
                            - intersect.mesh.bounds[0][2]
                            > 1e-3
                        ):
                            #     if node_name == 'frl_apartment_table_02_49':
                            #         print(node_name, stage_obj.name, 'left',EIGHT_DIRECTIONS[left_part], intersect.mesh.bounds)
                            intersect_2d_convex = intersect.cal_convex_hull_2d()
                            near_side = [rect[3], rect[2]]
                            far_side = [rect[0], rect[1]]
                            new_near_side, new_far_side = (
                                intersect_2d_convex.cut_free_space_with_point_cloud(
                                    near_side,
                                    far_side,
                                    node_bbox_8_directions[left_part],
                                    force=left_part % 2 == 0,
                                )
                            )
                            self.nodes[node_name].free_space[left_part][
                                "Critical_space"
                            ] = [
                                new_far_side[0],
                                new_far_side[1],
                                new_near_side[1],
                                new_near_side[0],
                            ]
                            #      if node_name == 'frl_apartment_table_02_49':
                            #           print('near',near_side,'far', far_side)
                            #          print('newnear', new_near_side,'newfar', new_far_side)
                            min_intersection_height = min(
                                min_intersection_height, intersect.mesh.bounds[0][2]
                            )
                    for right_part in range(5, 8):
                        rect = node.free_space[right_part]["Critical_space"]
                        if isinstance(rect, str):
                            continue

                        cuboid_vertices = [
                            [rect[0][0], rect[0][1], bottom],
                            [rect[1][0], rect[1][1], bottom],
                            [rect[2][0], rect[2][1], bottom],
                            [rect[3][0], rect[3][1], bottom],
                            [rect[0][0], rect[0][1], top],
                            [rect[1][0], rect[1][1], top],
                            [rect[2][0], rect[2][1], top],
                            [rect[3][0], rect[3][1], top],
                        ]
                        cuboid = MeshProcessor.create_cuboid_from_vertices(
                            cuboid_vertices
                        )
                        intersect = geometry.intersection_with_cuboid(cuboid)
                        #  if node_name == 'frl_apartment_table_04_14':
                        #         print(node_name, stage_obj.name, EIGHT_DIRECTIONS[right_part], intersect.mesh.bounds if intersect is not None else None)

                        if (
                            intersect is not None
                            and intersect.mesh.bounds[1][2]
                            - intersect.mesh.bounds[0][2]
                            > 1e-3
                        ):
                            intersect_2d_convex = intersect.cal_convex_hull_2d()
                            near_side = [rect[0], rect[1]]
                            far_side = [rect[3], rect[2]]
                            new_near_side, new_far_side = (
                                intersect_2d_convex.cut_free_space_with_point_cloud(
                                    near_side,
                                    far_side,
                                    node_bbox_8_directions[right_part],
                                    force=right_part % 2 == 0,
                                )
                            )
                            self.nodes[node_name].free_space[right_part][
                                "Critical_space"
                            ] = [
                                new_near_side[0],
                                new_near_side[1],
                                new_far_side[1],
                                new_far_side[0],
                            ]
                            min_intersection_height = min(
                                min_intersection_height, intersect.mesh.bounds[0][2]
                            )
                    for front_part in range(3, 6):
                        rect = node.free_space[front_part]["Critical_space"]
                        if isinstance(rect, str):
                            continue

                        cuboid_vertices = [
                            [rect[0][0], rect[0][1], bottom],
                            [rect[1][0], rect[1][1], bottom],
                            [rect[2][0], rect[2][1], bottom],
                            [rect[3][0], rect[3][1], bottom],
                            [rect[0][0], rect[0][1], top],
                            [rect[1][0], rect[1][1], top],
                            [rect[2][0], rect[2][1], top],
                            [rect[3][0], rect[3][1], top],
                        ]
                        cuboid = MeshProcessor.create_cuboid_from_vertices(
                            cuboid_vertices
                        )
                        intersect = geometry.intersection_with_cuboid(cuboid)
                        if (
                            intersect is not None
                            and intersect.mesh.bounds[1][2]
                            - intersect.mesh.bounds[0][2]
                            > 1e-3
                        ):
                            #      if node_name == 'frl_apartment_table_02_49':
                            #          print(node_name, stage_obj.name, 'front', EIGHT_DIRECTIONS[front_part], intersect.mesh.bounds)
                            intersect_2d_convex = intersect.cal_convex_hull_2d()
                            near_side = [rect[0], rect[3]]
                            far_side = [rect[1], rect[2]]
                            new_near_side, new_far_side = (
                                intersect_2d_convex.cut_free_space_with_point_cloud(
                                    near_side,
                                    far_side,
                                    node_bbox_8_directions[front_part],
                                    force=front_part % 2 == 0,
                                )
                            )
                            self.nodes[node_name].free_space[front_part][
                                "Critical_space"
                            ] = [
                                new_near_side[0],
                                new_far_side[0],
                                new_far_side[1],
                                new_near_side[1],
                            ]
                            #      if node_name == 'frl_apartment_table_02_49':
                            #          print(near_side, far_side, new_near_side, new_far_side)
                            min_intersection_height = min(
                                min_intersection_height, intersect.mesh.bounds[0][2]
                            )
                    #    else:
                    #        print(None if intersect is None else intersect.mesh.bounds)
                    for rear_part in [7, 0, 1]:
                        rect = node.free_space[rear_part]["Critical_space"]
                        if isinstance(rect, str):
                            continue

                        cuboid_vertices = [
                            [rect[0][0], rect[0][1], bottom],
                            [rect[1][0], rect[1][1], bottom],
                            [rect[2][0], rect[2][1], bottom],
                            [rect[3][0], rect[3][1], bottom],
                            [rect[0][0], rect[0][1], top],
                            [rect[1][0], rect[1][1], top],
                            [rect[2][0], rect[2][1], top],
                            [rect[3][0], rect[3][1], top],
                        ]
                        cuboid = MeshProcessor.create_cuboid_from_vertices(
                            cuboid_vertices
                        )
                        intersect = geometry.intersection_with_cuboid(cuboid)
                        if (
                            intersect is not None
                            and intersect.mesh.bounds[1][2]
                            - intersect.mesh.bounds[0][2]
                            > 1e-3
                        ):
                            #   if node_name == 'frl_apartment_table_04_14':
                            #       print(node_name, stage_obj.name, EIGHT_DIRECTIONS[rear_part], intersect.mesh.bounds)
                            intersect_2d_convex = intersect.cal_convex_hull_2d()
                            near_side = [rect[1], rect[2]]
                            far_side = [rect[0], rect[3]]
                            new_near_side, new_far_side = (
                                intersect_2d_convex.cut_free_space_with_point_cloud(
                                    near_side,
                                    far_side,
                                    node_bbox_8_directions[rear_part],
                                    force=rear_part % 2 == 0,
                                )
                            )
                            self.nodes[node_name].free_space[rear_part][
                                "Critical_space"
                            ] = [
                                new_far_side[0],
                                new_near_side[0],
                                new_near_side[1],
                                new_far_side[1],
                            ]
                            min_intersection_height = min(
                                min_intersection_height, intersect.mesh.bounds[0][2]
                            )
        #                self.nodes[node_name].free_space_height[1] = min(node.free_space_height[1], min_intersection_height)

        for node_name, node in self.nodes.items():
            self.nodes[node_name].free_space_height[1] = min(
                self.nodes[node_name].free_space_height[1], max_stage_height
            )

    def fix_heading_for_all_ground_objects(self):
        for ground_object_name, ground_object in self.nodes.items():
            if ground_object.depth == 1:
                standable_heading = 0
                for i in range(0, 8, 2):
                    if ground_object.freespace_is_standable(i):
                        standable_heading = i
                        break
                self.nodes[ground_object_name].rotate_free_space_for_all_children(
                    standable_heading
                )

    def set_node(self, node):
        self.nodes[node.name] = node

    def get_node(self, name):
        return self.nodes.get(name, None)

    def remove_node_from_scene(self, name):
        node = self.get_node(name)
        if node is not None:
            for entity in self.corresponding_scene.entities:
                if node.name in entity.get_name():
                    entity.remove_from_scene()
                    break

    def add_node_to_scene(self, name):
        node = self.get_node(name)
        if node is not None:
            obj = node.entity_config
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

            builder = self.corresponding_scene.create_actor_builder()
            builder.add_visual_from_file(filename=object_file_path)
            if collision_path is not None:
                builder.add_multiple_convex_collisions_from_file(
                    filename=collision_path
                )
            else:
                builder.add_convex_collision_from_file(filename=object_file_path)

            #    if obj["motion_type"] == "STATIC" or obj["motion_type"] == "KEEP_FIXED":
            #           mesh = builder.build_static(name=obj["name"])
            #      else:
            mesh = builder.build_static(name=obj["name"])
            mesh.set_pose(sapien.Pose(p=position, q=quaternion))

    def remove_node(self, name):
        node = self.get_node(name)

        if node is not None:
            print(
                "removing node",
                node.name,
                node.parent.name if node.parent is not None else None,
            )
            parent_name = node.parent.name if node.parent is not None else None
            self.nodes[name].removed = True
            if node.parent is not None:
                self.nodes[parent_name].remove_child(node)

            for dir in range(8):
                self.nodes[name].free_space[dir]["Objects"] = []

            self.dfs_for_freespace(self.nodes[parent_name])
            node.parent = None
            self.remove_node_from_scene(name)

    def add_node(self, node_name, parent_name, platform_name, translation):

        if parent_name not in self.nodes.keys():
            print("Parent node not found", parent_name, platform_name)
            return
        node = self.nodes[node_name]
        parent = self.nodes[parent_name]
        platform = None
        for plat in parent.own_platform:
            if plat.name == platform_name:
                platform = plat
                break
        if platform is None:
            print("Platform not found", parent_name, platform_name)
            return

        node.removed = False
        node.parent = parent
        node.on_platform = platform
        node.free_space_height = [
            platform.height,
            platform.height + platform.avl_height,
        ]

        node.heading = node.object.heading = node.object.convex_hull_2d.heading = (
            parent.heading
        )

        node.object.convex_hull_2d.vertices += translation

        new_bottom = platform.height + 0.01
        node.entity_config["centroid_translation"]["x"] += translation[0]
        node.entity_config["centroid_translation"]["y"] += translation[1]
        node.entity_config["centroid_translation"]["z"] += new_bottom - node.bottom

        node.top = node.top - node.bottom + new_bottom
        node.bottom = new_bottom
        # print(parent.display_unique_information(), node.display_unique_information(), platform.name)

        self.nodes[node_name] = node
        parent.add_child(self.nodes[node_name])
        #  print(parent.display_unique_information())
        #   print(node.display_unique_information())
        #  print(child.name for child in parent.children)
        self.dfs_for_freespace(self.nodes[parent_name])

        self.add_node_to_scene(node.name)
        self.nodes[node_name].display_unique_information()
        # for i in range(10):
        #            self.corresponding_scene.step()
        #            self.corresponding_scene.update_render()
        pass


def print_tree(node, depth=0):
    print("  " * depth + f"Object Name: {node.name}")

    all_free = True
    for direction in node.free_space.keys():
        if node.free_space[direction]["Empty"] == False:
            print(
                "  " * depth
                + f'*Occupied Direction: {direction}, Objects: {node.free_space[direction]["Objects"]}'
            )
            all_free = False
    if all_free:
        print("  " * depth + f"*All Directions are Free")

    for child in node.children:
        print_tree(child, depth + 1)


# Function to generate the tree structure starting from ground objects
# key function
def gen_multi_layer_graph_with_free_space(json_data):

    all_objects = [obj for obj in json_data["object_instances"]]

    scene_platform_list = []
    stage = [
        {
            "name": "stage",
            "visual_path": json_data["background_file_path"],
            "centroid_translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "bbox": "deprecated",
        }
    ]

    ground = parse_objects.SceneObject(name="GROUND")
    ground.set_ground()
    scene_platform_list.append(
        parse_objects.ScenePlatform(
            name="GROUND_0",
            heading=(1, 0),
            height=0,
            avl_height=3.08,
            bbox=[[-INF, -INF], [INF, INF]],
            instancetype="platform",
            convex_hull_2d=ConvexHullProcessor_2d(
                vertices=[[-INF, -INF], [INF, -INF], [INF, INF], [-INF, INF]],
                heading=(1, 0),
            ),
            belong="GROUND",
        )
    )
    object_list = parse_objects.create_object_list(
        all_objects, calculate_affordable_platforms=True
    )
    object_list.append(ground)

    stage_list = parse_objects.create_object_list(
        stage, calculate_affordable_platforms=False
    )
    for object in object_list:
        # the height of object is the height of the object's bottom.
        # Note: for object, get_bounding_box has already calculated the bbox_min and bbox_max in the world coordinate system, added the centroid_translation.
        bbox_min, bbox_max = object.get_bounding_box()
        scene_platform_list.append(
            parse_objects.ScenePlatform(
                name=object.name,
                heading=object.heading,
                height=bbox_min[2],
                top=bbox_max[2],
                bbox=[bbox_min[:2], bbox_max[:2]],
                bounding_points=object.bounding_points,
                convex_hull_2d=object.convex_hull_2d,
                instancetype="object",
            )
        )
        # the height of platform is the height of the platform's top.
        for geometry in object.mesh:
            for platform in geometry.affordable_platforms:
                scene_platform_list.append(
                    parse_objects.ScenePlatform(
                        name=object.name + platform.name,
                        heading=object.heading,
                        bbox=platform.bbox
                        + np.array(
                            [
                                object.centroid_translation[:2],
                                object.centroid_translation[:2],
                            ]
                        ),
                        height=platform.get_height()[1]
                        + object.centroid_translation[2],
                        avl_height=platform.available_height,
                        convex_hull_2d=platform.get_convex_hull_2d()
                        + object.centroid_translation[:2],
                        instancetype="platform",
                        belong=object.name,
                    )
                )
            # print(platform.get_convex_hull_2d() + object.centroid_translation[:2])

    sorted_scene_platform_list, contacts_id = (
        parse_objects.ScenePlatform.calculate_contact_conditions(scene_platform_list)
    )

    contacts = [
        (sorted_scene_platform_list[i[0]], sorted_scene_platform_list[i[1]])
        for i in contacts_id
    ]

    scene_graph_tree = Tree()
    scene_graph_tree.from_scene_platform_list(sorted_scene_platform_list, contacts)

    scene_graph_tree.calculate_free_space()
    # scene_graph_tree.clean_zero_area_free_space()
    scene_graph_tree.cut_free_space_with_stage(stage_list)

    scene_graph_tree.fix_heading_for_all_ground_objects()

    for obj_instance in all_objects:
        if scene_graph_tree.get_node(obj_instance["name"]) is None:
            print(obj_instance["name"], "not in the tree")
        scene_graph_tree.nodes[obj_instance["name"]].entity_config = obj_instance

    # scene_graph_tree.clean_zero_area_free_space()
    if "cabinet_3_body" in scene_graph_tree.nodes.keys():
        print(scene_graph_tree.nodes["cabinet_3_body"].heading)
        pass
    return scene_graph_tree


def main():
    # Load JSON data and generate the tree structure
    # input_file_path = './replica_apt_0.json'
    # input_file_path = './toy_scene_2.json'
    parser = argparse.ArgumentParser(
        description="Generate scene graph with free space information."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="entities_apt_0.json",
        required=False,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./result/replica_apt_0.txt",
        required=False,
        help="Path to the output text file.",
    )
    parser.add_argument(
        "--output_scene_graph_tree",
        type=str,
        default="./result/scene_graph_tree_0.txt",
        required=False,
        help="Path to the output text file.",
    )
    args = parser.parse_args()

    input_file_path = args.input
    output_file_path = args.output
    output_scene_graph_path = args.output_scene_graph_tree
    input_json = load_json_file(input_file_path)

    with open(output_file_path, "w") as output_file:
        output_file.write("----------------------------------\n")
        output_file.write("Establishing the tree structure:\n")
        result_tree = gen_multi_layer_graph_with_free_space(input_json)

        output_file.write("\n----------------------------------\n")
        output_file.write("Result tree:\n")
        for root_node in result_tree.nodes.values():
            if (
                root_node.parent is None
            ):  # Only print root nodes (those without a parent)
                output_file.write("---\n")
                output_file.write(f"Object Name: {root_node.name}\n")
                all_free = True
                if root_node.free_space is None:
                    output_file.write(f"*No Free Space Information\n")
                    continue
                for direction in range(len(root_node.free_space)):
                    if len(root_node.free_space[direction]["Objects"]):
                        output_file.write(
                            f'*Occupied Direction : {direction}, Objects: {root_node.free_space[direction]["Objects"]}\n'
                        )
                        all_free = False
                if all_free:
                    output_file.write(f"*All Directions are Free\n")
                output_file.write("---\n")

        output_file.write("\n----------------------------------\n")
        output_file.write("Node details in result tree:\n")
        for obj in input_json["object_instances"]:
            node = result_tree.get_node(obj["name"])
            if node is None:
                continue
            output_file.write("---\n")
            output_file.write(
                f"Name: {node.name}, Parent: {node.parent.name if node.parent else 'None'}, Children: {[child.name for child in node.children]}\n"
            )
            output_file.write(
                f"convex_bbox: {node.object.convex_hull_2d.get_headed_bbox_instance()}\n"
            )
            output_file.write(
                f"convex: {node.object.convex_hull_2d.get_vertices_on_convex_hull()}\n"
            )
            output_file.write(f"Heading: {node.heading}\n")
            output_file.write(f"Top: {node.top}\n")
            output_file.write(f"Belong to platform: {node.on_platform.name}\n")
            for direction in range(len(root_node.free_space)):
                free_space_info = node.free_space[direction]["Available_space"]
                critical_space_info = node.free_space[direction]["Critical_space"]

                output_file.write("-\n")
                output_file.write(
                    f"Direction: {EIGHT_DIRECTIONS[direction]},  Objects: {[object.name for object in node.free_space[direction]['Objects']]}\n"
                )
                output_file.write(f"Available Space: \n[{free_space_info}]\n")
                if isinstance(critical_space_info, str):
                    output_file.write(f"Critical Available Space: \nNot Available\n")
                else:
                    output_file.write(
                        f"Critical Available Space: \n[{critical_space_info}]\n"
                    )
                output_file.write(
                    f"Free Space Height: \n[{float(node.free_space_height[0])},{float(node.free_space_height[1])}]\n"
                )
            output_file.write("-\n")
            output_file.write(f"Own Platform: \n")
            for platform in node.own_platform:
                output_file.write(f"{platform.name}\n")
                output_file.write(
                    f"bbox: \n[[{platform.bbox[0][0]},{platform.bbox[0][1]}],[{platform.bbox[1][0]},{platform.bbox[1][1]}]] \n"
                )
                output_file.write(f"height: \n{platform.height}\n")
                output_file.write(f"available height: \n{platform.avl_height}\n")
            output_file.write("\n")

    visualize_tree(result_tree, file=open(output_scene_graph_path, "w"))


if __name__ == "__main__":
    main()


"""


"""
# %%
