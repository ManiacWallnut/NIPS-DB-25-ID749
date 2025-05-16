import sys
import os
from pygltflib import GLTF2, BufferFormat
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from typing import List
from shapely.geometry import Polygon, LinearRing, LineString, Point
import matplotlib.pyplot as plt
from .meshprocessor import MeshProcessor
from scipy.spatial.transform import Rotation as R
from .convex_hull_processor import ConvexHullProcessor_2d, Basic2DGeometry
from .point_sum_query_processor import PointSumQuery2D
import glog

# import affordable_face


def reindex(vertex_indices, old_faces):
    new_faces = []
    for face in old_faces:
        new_face = [
            np.where(vertex_indices == face[i])[0][0] for i in range(face.shape[0])
        ]
        new_faces.append(new_face)
    return new_faces


# either a affordable platform or a object.
# mainly use to sort all the object and platform according to
class ScenePlatform:
    CONTACT_EPS = 2.05e-2
    BBOX_EPS = 4e-1

    def __init__(
        self,
        name="",
        height=0,
        avl_height=0,
        top=0,
        bbox=[(0, 0), (1e9, 1e9)],
        bounding_points=[],
        convex_hull_2d=[],
        heading=(1, 0),
        instancetype="platform",
        belong=None,
    ):
        # height of the platform
        self.name = name
        self.height = height
        self.heading = heading
        self.avl_height = avl_height
        self.top = top
        self.bounding_points = bounding_points
        # if object, use bbox. but if platform, maybe need to use vertices and faces
        self.bbox = bbox
        # remember to put it in here.
        self.convex_hull_2d = convex_hull_2d
        # type (object or platform)
        if instancetype == "platform" or instancetype == "object":
            self.type = instancetype
        else:
            print("Warning: The type of the platform is unexpected.")
        # belong = none for objects, = object_name for platforms
        self.bel_object = belong

    def __repr__(self):
        return f"""
            ScenePlatform(
            name={self.name},
            height={self.height}, 
            bbox={self.bbox}, 
            heading={self.heading}, 
            type={self.type}, 
            belong={self.bel_object})
            """

    @staticmethod
    def sort_platforms(platforms):
        # sort the platforms according to the height, and with the same height, let the platform be in front of the platform
        return sorted(platforms, key=lambda x: x.height)

    @staticmethod
    def bbox_bbox_intersection(bbox1, bbox2):
        min1, max1 = bbox1
        min2, max2 = bbox2
        return (max(min1[0], min2[0]), max(min1[1], min2[1])), (
            min(max1[0], max2[0]),
            min(max1[1], max2[1]),
        )

    @staticmethod
    def bbox_bbox_intersection_area(bbox1, bbox2):
        bbox = ScenePlatform.bbox_bbox_intersection(bbox1, bbox2)
        return max(bbox[1][0] - bbox[0][0], 0) * max(bbox[1][1] - bbox[0][1], 0)

    @staticmethod
    def platform_bbox_intersection(platform, bbox):
        return platform.intersect_rectangle_area(bbox)

    # The following function still use bbox instead of convex_hull to calculate the contact conditions
    # But it works well, at least works well in replica.
    # Note that the logic like this cannot be use on the further stages such as generating the tasks.
    @staticmethod
    def calculate_contact_conditions(platforms):
        # sort platforms and objects, and use two pointers to find the contact conditions
        n = len(platforms)
        negative_height_adjustment = {}
        for i in range(n):
            if "GROUND" in platforms[i].name:
                continue
            if platforms[i].height < 0 and platforms[i].type == "object":
                negative_height_adjustment[platforms[i].name] = (
                    1e-4 - platforms[i].height
                )
                platforms[i].height = 1e-4
            elif (
                platforms[i].type == "platform"
                and platforms[i].bel_object in negative_height_adjustment
            ):
                platforms[i].height += negative_height_adjustment[
                    platforms[i].bel_object
                ]
        platforms = ScenePlatform.sort_platforms(platforms)
        # id_contacts should be a list of tuples, each tuple is a pair of indices(x,y), x is the index of the platform, y is the index of the object, and y is on top of x

        # Note:
        # For objects, we still use the whole bbox to calculate the contact conditions, but for platforms we use polygon to calculate the contact conditions.
        # It may cause some problems, in cases such as the object is a lamp with a huge head but a small base, and we may consider the head of the lamp as the contact condition.
        # But it can't be solved with simply calculate the platform with the lowest z, because there will be lanes and other platforms that are lower than the head of the lamp.
        id_contacts = []
        id_possible_contact = 0

        # for i in range(min(50,n)):
        #    print('calculate_contact_conditions', i, platforms[i].name, platforms[i].bel_object, platforms[i].type, platforms[i].height, platforms[i].bbox)

        for i in range(n):
            if platforms[i].name == "GROUND":
                continue

            if platforms[i].type == "object":
                id_possible_contact = i - 1

                while True:
                    if (
                        platforms[i].height - platforms[id_possible_contact].height
                        > ScenePlatform.CONTACT_EPS
                        or id_possible_contact < 0
                    ):
                        id_possible_contact = i + 1

                    if (
                        platforms[id_possible_contact].height - platforms[i].height
                        > ScenePlatform.CONTACT_EPS
                        and id_possible_contact >= i
                    ):
                        break

                    if (
                        platforms[id_possible_contact].type == "object"
                        or platforms[id_possible_contact].bel_object
                        == platforms[i].name
                    ):
                        id_possible_contact -= 1 if id_possible_contact < i else -1
                        continue

                    #   object_area = platforms[i].convex_hull_2d.get_area()

                    object_area = min(
                        platforms[i].convex_hull_2d.get_area(),
                        platforms[id_possible_contact].convex_hull_2d.get_area(),
                    )
                    bbox_intersect_area = platforms[
                        i
                    ].convex_hull_2d.intersect_area_with_another_convex(
                        platforms[id_possible_contact].convex_hull_2d
                    )
                    # print(object_area, bbox_intersect_area, platforms[i].name, platforms[id_possible_contact].name)

                    # object_area = min((platforms[i].bbox[1][0] - platforms[i].bbox[0][0]) * (platforms[i].bbox[1][1] - platforms[i].bbox[0][1]),
                    #  (platforms[id_possible_contact].bbox[1][0] - platforms[id_possible_contact].bbox[0][0]) * (platforms[id_possible_contact].bbox[1][1] - platforms[id_possible_contact].bbox[0][1]))
                    # bbox_intersect_area = ScenePlatform.bbox_bbox_intersection_area(platforms[id_possible_contact].bbox, platforms[i].bbox)

                    # print(object_area, bbox_intersect_area, platforms[i].name, platforms[id_possible_contact].name)

                    # print('bbox_intersect_area', ScenePlatform.bbox_bbox_intersection(platforms[id_possible_contact].bbox, platforms[i].bbox), 'object_area', object_area, 'i', i, 'id', id_possible_contact)
                    if bbox_intersect_area > object_area * ScenePlatform.BBOX_EPS:
                        id_contacts.append((id_possible_contact, i))
                        break
                    else:
                        id_possible_contact -= 1 if id_possible_contact < i else -1
                        if bbox_intersect_area > 1e-6:
                            print(
                                "warning: Too small bbox_intersect_area",
                                bbox_intersect_area,
                                "object_area",
                                object_area,
                                "i",
                                platforms[i].name,
                                "id",
                                platforms[id_possible_contact].name,
                            )

        pass

        return platforms, id_contacts

    def find_available_places(self, obstacle_list, target, min_step=0.005):
        # obstacle_list 是一个SceneBox的list.
        # target 是一个SceneBox
        # min_step 是一个float
        # 返回一系列的位置，这些位置可以放置target，但是不会和obstacle_list中的任何一个box相交。
        table_vertices = self.bbox
        rotation_angle = np.arctan2(self.heading[1], self.heading[0])

        transformed_table = np.array(
            [
                Basic2DGeometry.rotate_points(table_vertice, rotation_angle)
                for table_vertice in table_vertices
            ]
        )

        #        glog.info(f"transformed_table: {transformed_table}")

        transformed_obstacles = np.array(
            [
                np.array(
                    [
                        Basic2DGeometry.rotate_points(obstacle_vertice, rotation_angle)
                        for obstacle_vertice in obstacle
                    ]
                )
                for obstacle in obstacle_list
            ]
        )

        #        glog.info(f"transformed_obstacles: {transformed_obstacles}")

        transformed_target = np.array(
            [
                Basic2DGeometry.rotate_points(target_vertice, rotation_angle)
                for target_vertice in target
            ]
        )

        x_min, x_max, y_min, y_max = (
            np.min(transformed_table[:, 0]),
            np.max(transformed_table[:, 0]),
            np.min(transformed_table[:, 1]),
            np.max(transformed_table[:, 1]),
        )
        target_x_min, target_x_max, target_y_min, target_y_max = (
            np.min(transformed_target[:, 0]),
            np.max(transformed_target[:, 0]),
            np.min(transformed_target[:, 1]),
            np.max(transformed_target[:, 1]),
        )
        target_x_len, target_y_len = (
            target_x_max - target_x_min,
            target_y_max - target_y_min,
        )

        prefix_points, prefix_values = [], []
        sensible_xs, sensible_ys = [], []

        for obs in transformed_obstacles:
            obs_x_min, obs_x_max, obs_y_min, obs_y_max = (
                np.min(obs[:, 0]),
                np.max(obs[:, 0]),
                np.min(obs[:, 1]),
                np.max(obs[:, 1]),
            )
            obs_x_min = np.floor(obs_x_min / min_step) * min_step
            obs_y_min = np.floor(obs_y_min / min_step) * min_step
            obs_x_max = np.ceil(obs_x_max / min_step) * min_step
            obs_y_max = np.ceil(obs_y_max / min_step) * min_step

            x = obs_x_min
            y = obs_y_min
            while x < obs_x_max:
                while y < obs_y_max:
                    prefix_points.append((x, y))
                    prefix_values.append(1)
                    y += min_step
                x += min_step
                y = obs_y_min

            sensible_xs.append(obs_x_min)
            sensible_xs.append(obs_x_max)
            sensible_ys.append(obs_y_min)
            sensible_ys.append(obs_y_max)

        point_sum_query = PointSumQuery2D(prefix_points, prefix_values)

        sensible_xs.extend([x_min, x_max])
        sensible_ys.extend([y_min, y_max])

        available_positions = []

        for op in range(4):
            op_offset = transformed_target[0] - transformed_target[op]
            for x in sensible_xs:
                for y in sensible_ys:
                    x1, x2, y1, y2 = x, x + target_x_len, y, y + target_y_len
                    x1, x2, y1, y2 = (
                        x1 + op_offset[0],
                        x2 + op_offset[0],
                        y1 + op_offset[1],
                        y2 + op_offset[1],
                    )
                    if x1 < x_min or x2 > x_max or y1 < y_min or y2 > y_max:
                        continue
                    if point_sum_query.query_rect(x1, y1, x2, y2) <= 0:
                        available_positions.append(
                            [
                                np.array([x1, y1]),
                                np.array([x1, y2]),
                                np.array([x2, y2]),
                                np.array([x2, y1]),
                            ]
                        )

        available_positions = [
            [
                Basic2DGeometry.rotate_points(vertice, -rotation_angle)
                for vertice in available_rect
            ]
            for available_rect in available_positions
        ]

        #   glog.info(f"available_positions: {available_positions}")

        return available_positions

    def get_all_available_freespace_combinations(
        self, object_list, target, min_step=0.005
    ):

        obstacle_bbox_list = [obj.object.bbox for obj in object_list]
        target_bbox = target.bbox
        available_positions = self.find_available_places(
            obstacle_bbox_list, target_bbox, min_step
        )

        possible_freespace_combination = {}

        if len(obstacle_bbox_list) == 0:
            possible_freespace_combination[-1] = [[0, 0], available_positions]
            return possible_freespace_combination

        in_freespace_id_hash = 0

        for available_rect in available_positions:
            assert len(available_rect) == 4
            in_freespace_id = []
            in_freespace_id_hash = 0

            for id, object in enumerate(object_list):
                for dir in range(8):
                    rect_vertice_inside_rectangle = [
                        Basic2DGeometry.is_inside_rectangle(
                            rect_vertice, object.free_space[dir]["Critical_space"]
                        )
                        for rect_vertice in available_rect
                    ]
                    if all(rect_vertice_inside_rectangle):
                        in_freespace_id_hash = (
                            in_freespace_id_hash * 100 + id * 10 + dir
                        ) % 1000000007
                        in_freespace_id.append([id, dir])
            if in_freespace_id_hash not in possible_freespace_combination:
                possible_freespace_combination[in_freespace_id_hash] = [
                    in_freespace_id,
                    available_rect,
                ]

        return possible_freespace_combination


class SceneBox:
    bbox_margin_adjustment = 1e-2

    def __init__(
        self, heading=(1, 0), bbox=[(0, 0), (0, 0)], convex_hull_2d=None, name=""
    ):
        self.heading = heading
        self.name = name
        self.convex_hull_2d = convex_hull_2d
        self.convex_hull_2d.heading = heading
        self.bbox = self.convex_hull_2d.get_headed_bbox_instance()
        pass

    def is_beyond_surface_bbox(self, other_object, heading=(1, 0)):

        box_left_side = [self.bbox[0], self.bbox[1]]
        box_right_side = [self.bbox[3], self.bbox[2]]
        box_front_side = [self.bbox[1], self.bbox[2]]
        box_rear_side = [self.bbox[0], self.bbox[3]]

        l, r, lr, u, d, ud = False, False, False, False, False, False

        for point in other_object.bbox:
            left = np.cross(
                box_left_side[1] - box_left_side[0], point - box_left_side[0]
            )
            right = np.cross(
                box_right_side[1] - box_right_side[0], point - box_right_side[0]
            )
            front = np.cross(
                box_front_side[1] - box_front_side[0], point - box_front_side[0]
            )
            rear = np.cross(
                box_rear_side[1] - box_rear_side[0], point - box_rear_side[0]
            )
            l |= left > 0 and right > 0
            lr |= left < 0 and right > 0
            r |= left < 0 and right < 0
            u |= front > 0 and rear > 0
            ud |= front < 0 and rear > 0
            d |= front < 0 and rear < 0

        at_directions = []

        if l and u:
            at_directions.append("front-left")
        if l and d:
            at_directions.append("rear-left")
        if r and u:
            at_directions.append("front-right")
        if r and d:
            at_directions.append("rear-right")

        # if l and u and d, the object is occupied the whole left side of the surface;
        # if l and ud, the object  is occupied the exact left side of the surface and/or the front-left / rear-left side of the surface
        if l and ((u and d) or ud):
            at_directions.append("left")
        if r and ((u and d) or ud):
            at_directions.append("right")
        if u and ((l and r) or lr):
            at_directions.append("front")
        if d and ((l and r) or lr):
            at_directions.append("rear")

        return at_directions


class SceneObject:
    def __init__(
        self,
        heading=(1, 0),
        geometries=[],
        derivatives=[],
        centroid_translation=(0, 0, 0),
        quaternion=(1, 0, 0, 0),
        rpy=(0, 0, 0),
        bbox=(0, 0, 0),
        name="",
    ):
        # heading, should be a unit vector xy. Usually it should be either of 8 directions.
        self.heading = heading
        # list of geometries. TVstand has 3, other has 1 in replica_01.
        self.mesh = []

        self.eps = 1e-6

        self.derivatives = derivatives
        self.bounding_points = []
        # bbox center
        self.centroid_translation = centroid_translation
        # maybe it need to be deprecated
        self.quaternion = quaternion
        # roll, pitch, yaw
        self.rpy = rpy
        if rpy[0] != 0 or rpy[1] != 0 or rpy[2] != 0:
            self.quaternion = R.from_euler("xyz", rpy).as_quat()
        elif (
            quaternion[0] != 0
            or quaternion[1] != 0
            or quaternion[2] != 0
            or quaternion[3] != 0
        ):
            self.rpy = R.from_quat(quaternion).as_euler("xyz")
        else:
            print("Warning: No rotation information is provided.")

        # bbox
        self.bbox = self.get_bounding_box()
        self.height = self.bbox[0][2]

        # name
        self.name = name

        vertices = []
        for geometry in geometries:
            self.mesh.append(MeshProcessor(geometry))
            vertices.extend(geometry.vertices)
        vertices = np.array(vertices)
        if len(vertices):
            vertices += np.array(
                [
                    centroid_translation[0],
                    centroid_translation[1],
                    centroid_translation[2],
                ]
            )

            self.convex_hull_2d = ConvexHullProcessor_2d(vertices, heading)

        # constants

    def set_ground(self):
        ground_vertices = np.array(
            [
                [-1e2, -1e2, 0],
                [1e2, -1e2, 0],
                [1e2, 1e2, 0],
                [-1e2, 1e2, 0],
                [-1e2, -1e2, -1e-4],
                [1e2, -1e2, -1e-4],
                [1e2, 1e2, -1e-4],
                [-1e2, 1e2, -1e-4],
            ]
        )

        ground_faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom face
                [4, 5, 6],
                [4, 6, 7],  # top face
                [0, 1, 5],
                [0, 5, 4],  # front face
                [1, 2, 6],
                [1, 6, 5],  # right face
                [2, 3, 7],
                [2, 7, 6],  # back face
                [3, 0, 4],
                [3, 4, 7],  # left face
            ]
        )

        ground_mesh = trimesh.Trimesh(vertices=ground_vertices, faces=ground_faces)
        self.mesh = [MeshProcessor(ground_mesh)]
        self.convex_hull_2d = ConvexHullProcessor_2d(ground_vertices[:, :2], (1, 0))

        return None

    @property
    def base_threshold(self):
        return self.platform_base_threshold

    @property
    def adjacent_threshold(self):
        return self.platform_adjacent_threshold

    @property
    def absolute_threshold(self):
        return self.platform_absolute_threshold

    def get_bounding_box(self):
        cx, cy, cz = self.centroid_translation
        min_bbox = np.array([np.inf, np.inf, np.inf])
        max_bbox = np.array([-np.inf, -np.inf, -np.inf])
        for mesh in self.mesh:
            bbox_min, bbox_max = mesh.get_bounding_box()
            vertices = np.array(
                [
                    [bbox_min[0], bbox_min[1], bbox_min[2]],
                    [bbox_max[0], bbox_min[1], bbox_min[2]],
                    [bbox_max[0], bbox_max[1], bbox_min[2]],
                    [bbox_min[0], bbox_max[1], bbox_min[2]],
                    [bbox_min[0], bbox_min[1], bbox_max[2]],
                    [bbox_max[0], bbox_min[1], bbox_max[2]],
                    [bbox_max[0], bbox_max[1], bbox_max[2]],
                    [bbox_min[0], bbox_max[1], bbox_max[2]],
                ]
            )
            # rotation = R.from_euler('xyz', self.rpy)
            # vertices = rotation.apply(vertices)
            min_bbox = np.minimum(vertices.min(axis=0), min_bbox)
            max_bbox = np.maximum(vertices.max(axis=0), max_bbox)
        min_bbox += np.array([cx, cy, cz])
        max_bbox += np.array([cx, cy, cz])
        return min_bbox, max_bbox

    def cal_heading(self):
        volume = -1
        for geometry in self.mesh:
            bounding_points, orientation = geometry.cal_orientation()
            if abs(orientation[0]) < 1e-4:
                orientation = (0, orientation[1])
            if abs(orientation[1]) < 1e-4:
                orientation = (orientation[0], 0)
            if volume < len(geometry.mesh.vertices):
                self.heading = orientation
                self.bounding_points = bounding_points  #
        self.bounding_points += np.array(
            [self.centroid_translation[0], self.centroid_translation[1]]
        )
        #  print(self.name, self.heading)
        return None

    def cal_convex_hull_2d(self):
        vertices = []
        for geometry in self.mesh:
            vertices.extend(geometry.mesh.vertices)

        self.convex_hull_2d = ConvexHullProcessor_2d(vertices, self.heading)
        self.convex_hull_2d += np.array(
            [self.centroid_translation[0], self.centroid_translation[1]]
        )

    def get_2d_bbox(self):
        bbox_min, bbox_max = self.get_bounding_box()
        return bbox_min[:2], bbox_max[:2]

    def repair_object(self):
        self.mesh = [mesh.mesh_after_merge_close_vertices() for mesh in self.mesh]
        self.mesh = [mesh.repair_mesh_instance() for mesh in self.mesh]

    def is_on_top_of_surface(self, other_object):
        min_self, max_self = self.get_bounding_box()
        min_other, max_other = other_object.get_bounding_box()

        other_object_x = other_object.centroid_translation[0]
        other_object_y = other_object.centroid_translation[1]

        tz_top = max_self[2]
        oz_bottom = min_self[2]

        in_contact_z = abs(tz_top - oz_bottom) < self.eps

        within_x = min_self[0] < other_object_x and other_object_x < max_self[0]
        within_y = min_self[1] < other_object_y and other_object_y < max_self[1]

        return in_contact_z and within_x and within_y

    def is_on_supportable_plane(self, plane_height=0):
        min_bbox, max_bbox = self.get_bounding_box()
        return abs(min_bbox[2] - plane_height) < self.eps

    def is_bottom_on_same_height(self, other_object):
        min_self, max_self = self.get_bounding_box()
        min_other, max_other = other_object.get_bounding_box()

        self_bottom = min_self[2]
        other_bottom = min_other[2]

        return abs(self_bottom - other_bottom) < self.eps

    def calculate_affordable_platforms(self):
        new_derivatives = []
        max_area = 0
        for mesh in self.mesh:
            mesh.get_raw_affordable_platforms()
            if len(mesh.affordable_platforms) == 0:
                continue
            max_area = max(
                max_area,
                np.max([platform.get_area() for platform in mesh.affordable_platforms]),
            )
        for mesh in self.mesh:
            mesh.affordable_platforms = [
                platform
                for platform in mesh.affordable_platforms
                if platform.get_area() > max_area * 0.25
            ]

        for mesh in self.mesh:
            new_derivatives.extend(mesh.calculate_affordable_platforms())
        return new_derivatives

    def compute_overlap_bbox1_bbox2(self, other):
        min_self, max_self = self.get_bounding_box()
        min_other, max_other = other.get_bounding_box()

        x_min_overlap = max(min_self[0], min_other[0])
        x_max_overlap = min(max_self[0], max_other[0])
        y_min_overlap = max(min_self[1], min_other[1])
        y_max_overlap = min(max_self[1], max_other[1])

        x_min_overlap = min(x_min_overlap, x_max_overlap)
        y_min_overlap = min(y_min_overlap, y_max_overlap)

        return (x_min_overlap, y_min_overlap, x_max_overlap, y_max_overlap)

    def get_affordable_platforms(self):
        res = []
        for mesh in self.mesh:
            res.extend(mesh.get_affordable_platforms())
        return res


def create_object_list(object_mesh_list, calculate_affordable_platforms=True):
    objects = []
    idx = 0
    for mesh in object_mesh_list:
        path_prefix = "d:/workplace/scene_graph/task_generation/scene_graph/"
        path_prefix = "/home/weikang/Workspace/task-generation/scene_graph"
        # obj_config_path = Path(f"/media/iaskbd-ubuntu/E470A7DC9B7152FB/workplace/scene_graph/task_generation/scene_graph/scene_datasets/replica_cad_dataset/configs/objects/{template_name}.object_config.json")
        path_prefix = (
            "/media/iaskbd/7107462746C3F786/workplace/task_generation/scene_graph/"
        )
        if mesh["name"] == "ground" or mesh["name"] == "scene_background":
            continue
        glb = trimesh.load(os.path.join(path_prefix, mesh["visual_path"]))
        geometries = []
        centroid_translation = (
            mesh["centroid_translation"]["x"],
            mesh["centroid_translation"]["y"],
            mesh["centroid_translation"]["z"],
        )

        quaternion = (
            mesh["quaternion"]["w"],
            mesh["quaternion"]["x"],
            mesh["quaternion"]["y"],
            mesh["quaternion"]["z"],
        )

        # print(os.path.join(path_prefix, mesh['visual_path']))
        for mesh_name, geometry in glb.geometry.items():

            name = [node for node in glb.graph.nodes]

            name = glb.graph.geometry_nodes[mesh_name][0]

            if calculate_affordable_platforms == False:
                if not ("wall" in name or "blind" in name):
                    continue

            transform_matrix = np.array(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            )
            global_matrix = np.array(glb.graph[name][0])
            if (
                "kitchenCupboard_01" in mesh["name"]
                or "chestOfDrawers_01" in mesh["name"]
            ):
                for i in range(3):
                    for j in range(3):
                        global_matrix[i][j] /= 2.5

            quaternion_matrix = trimesh.transformations.quaternion_matrix(quaternion)

            geometry.apply_transform(transform_matrix @ global_matrix)

            geometry.apply_transform(quaternion_matrix)

            if calculate_affordable_platforms == False:
                new_glb = trimesh.Trimesh(
                    vertices=geometry.vertices, faces=geometry.faces
                )
                if new_glb.is_volume == False:
                    new_glb.vertices += np.random.rand(*new_glb.vertices.shape) * 1e-6
                    components = new_glb.split(only_watertight=False)
                    #  print(name)
                    for i in range(len(components)):
                        if components[i].is_volume == False:
                            component = components[i]
                            #       component.export(f"{name}_nontight_component_{i}.obj")
                            components[i] = MeshProcessor.repair_mesh(components[i])
                            if len(components[i].vertices) < 4:
                                continue
                            if components[i].is_volume == False:
                                components[i].vertices += (
                                    np.random.rand(*components[i].vertices.shape) * 1e-3
                                )
                                components[i] = components[i].convex_hull
                                if components[i].is_volume == False:
                                    continue
                        #        else:
                        #           components[i].export(f"{name}_tight_component_{i}.obj")
                        geometries.append(components[i])
                #       print(component.bounds)
                # geometries.extend(components)
            else:

                geometries.append(geometry)
            #  break

        idx += 1

        if len(geometries) > 1 and "tvstand" in mesh["name"]:
            # Combine all geometries into a single mesh if there are multiple geometries
            combined_vertices = []
            combined_faces = []
            vertex_offset = 0

            for geometry in geometries:
                combined_vertices.extend(geometry.vertices)
                combined_faces.extend(geometry.faces + vertex_offset)
                vertex_offset += len(geometry.vertices)

            combined_vertices = np.array(combined_vertices)
            combined_faces = np.array(combined_faces)

            # Create a single mesh from combined vertices and faces
            combined_mesh = trimesh.Trimesh(
                vertices=combined_vertices, faces=combined_faces
            )

            # Compute the convex hull of the combined mesh
            convex_hull = combined_mesh.convex_hull

            geometries = [
                trimesh.Trimesh(vertices=convex_hull.vertices, faces=convex_hull.faces)
            ]

        rpy = trimesh.transformations.euler_from_quaternion(quaternion, axes="sxyz")
        # print(mesh["name"])
        new_object = SceneObject(
            geometries=geometries,
            centroid_translation=centroid_translation,
            quaternion=quaternion,
            rpy=rpy,
            name=mesh["name"],
        )
        new_object.repair_object()
        new_object.cal_heading()
        new_object.cal_convex_hull_2d()
        objects.append(new_object)
        if calculate_affordable_platforms:
            potential_derivatives = new_object.calculate_affordable_platforms()

            if len(potential_derivatives) > 0:
                objects.extend(potential_derivatives)

        pass

    return objects


def main():
    path_prefix = "D:/workplace/scene_graph/task_generation/task_generation/"
    glb = trimesh.load(
        os.path.join(
            path_prefix,
            "scene_datasets/replica_cad_dataset/objects/frl_apartment_stool_02.glb",
        )
    )
    geometries = list(glb.geometry.values())
    for i in range(len(geometries)):
        transform_matrix = np.array(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
        geometries[i].apply_transform(transform_matrix)
    quaternion = (0.7071067811865476, 0, 0, 0.7071067811865476)
    rpy = trimesh.transformations.euler_from_quaternion(quaternion, axes="sxyz")
    new_object = SceneObject(
        geometries=geometries,
        centroid_translation=(0, 0, 0),
        quaternion=quaternion,
        rpy=rpy,
        name="tvstand",
    )
    new_object.repair_object()
    potential_derivatives = new_object.calculate_affordable_platforms()


if __name__ == "__main__":
    main()
