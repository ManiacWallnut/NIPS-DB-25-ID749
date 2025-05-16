from typing import List
import numpy as np
from typing import List
from shapely.geometry import Polygon, LinearRing, LineString, Point
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class ConvexHullProcessor_2d:
    def __init__(self, vertices: List[List[float]], heading=(1, 0)):
        self.vertices = np.array(vertices)
        if self.vertices.shape[1] > 2:
            self.vertices = self.vertices[:, :2]
        # convexhull.vertices is the index of the vertices on the convex hull
        # for 2d hull, convexhull.volume is the area of the hull, convexhull.area is the perimeter of the hull

        self.convex_hull = ConvexHull(self.vertices, qhull_options="QJ")

        self.heading = np.array(heading)

    def __repr__(self):
        return f"ConvexHullProcessor_2d({self.vertices}),{self.heading}"

    def plot_convex_hull(self, ax):
        convex_hull_vertices = self.get_vertices_on_convex_hull()
        convex_hull_vertices = np.vstack(
            [convex_hull_vertices, convex_hull_vertices[0]]
        )
        ax.plot(convex_hull_vertices[:, 0], convex_hull_vertices[:, 1], "r-")
        ax.plot(self.vertices[:, 0], self.vertices[:, 1], "bo")
        ax.set_aspect("equal")
        plt.show()

    def get_vertices_on_convex_hull(self):
        return self.vertices[self.convex_hull.vertices]

    def __add__(self, offset_2d):
        offset = np.array(offset_2d)
        return ConvexHullProcessor_2d(self.vertices + offset)

    def get_area(self):
        return self.convex_hull.volume

    @staticmethod
    def get_bbox(vertices):
        return np.min(vertices, axis=0)[:2], np.max(vertices, axis=0)[:2]

    @staticmethod
    def get_headed_bbox(vertices, heading=(1, 0)):
        heading = np.array(heading)
        perp_heading = np.array([-heading[1], heading[0]])

        min_proj_heading = np.inf
        max_proj_heading = -np.inf
        min_proj_perp_heading = np.inf
        max_proj_perp_heading = -np.inf

        for vertex in vertices:
            proj_heading = float(np.dot(vertex, heading))
            proj_perp_heading = float(np.dot(vertex, perp_heading))

            if proj_heading < min_proj_heading:
                min_proj_heading = proj_heading
            if proj_heading > max_proj_heading:
                max_proj_heading = proj_heading
            if proj_perp_heading < min_proj_perp_heading:
                min_proj_perp_heading = proj_perp_heading
            if proj_perp_heading > max_proj_perp_heading:
                max_proj_perp_heading = proj_perp_heading

        bbox = [
            min_proj_heading * heading + min_proj_perp_heading * perp_heading,
            min_proj_heading * heading + max_proj_perp_heading * perp_heading,
            max_proj_heading * heading + max_proj_perp_heading * perp_heading,
            max_proj_heading * heading + min_proj_perp_heading * perp_heading,
        ]

        return np.array(bbox)

    def can_fit_in(self, rect, allow_rotate=False):
        rect = np.array(rect)
        rect_heading = rect[1] - rect[0]
        rect_width, rect_height = (rect[1] - rect[0]).dot(rect[1] - rect[0]), (
            rect[2] - rect[1]
        ).dot(rect[2] - rect[1])
        rect_heading = rect_heading / np.linalg.norm(rect_heading)
        bbox_in_rect = ConvexHullProcessor_2d.get_headed_bbox(
            self.vertices, rect_heading
        )
        bbox_width, bbox_height = (bbox_in_rect[1] - bbox_in_rect[0]).dot(
            bbox_in_rect[1] - bbox_in_rect[0]
        ), (bbox_in_rect[2] - bbox_in_rect[1]).dot(bbox_in_rect[2] - bbox_in_rect[1])
        return bbox_width <= rect_width and bbox_height <= rect_height

    def get_fit_in_translation(self, rect):
        rect = np.array(rect)
        rect_heading = rect[1] - rect[0]
        rect_width, rect_height = (rect[1] - rect[0]).dot(rect[1] - rect[0]), (
            rect[2] - rect[1]
        ).dot(rect[2] - rect[1])
        rect_heading = rect_heading / np.linalg.norm(rect_heading)
        bbox_in_rect = ConvexHullProcessor_2d.get_headed_bbox(
            self.vertices, rect_heading
        )
        bbox_width, bbox_height = (bbox_in_rect[1] - bbox_in_rect[0]).dot(
            bbox_in_rect[1] - bbox_in_rect[0]
        ), (bbox_in_rect[2] - bbox_in_rect[1]).dot(bbox_in_rect[2] - bbox_in_rect[1])

        if bbox_width > rect_width or bbox_height > rect_height:
            return None

        bbox_center = (bbox_in_rect[0] + bbox_in_rect[2]) / 2
        rect_center = (rect[0] + rect[2]) / 2
        if np.isnan(bbox_center).any() or np.isnan(rect_center).any():
            return None
        return rect_center - bbox_center

    def get_bbox_instance(self):
        return ConvexHullProcessor_2d.get_bbox(self.vertices)

    def get_headed_bbox_instance(self):
        return ConvexHullProcessor_2d.get_headed_bbox(
            vertices=self.vertices, heading=self.heading
        )

    def get_closest_point_to_line(self, segment, default_point=None):
        convex_hull_vertices = self.get_vertices_on_convex_hull()
        closest_point = default_point
        min_distance = (
            Basic2DGeometry.point_to_line_distance(closest_point, segment)
            if default_point is not None
            else 1e9
        )
        for vertex in convex_hull_vertices:
            distance = Basic2DGeometry.point_to_line_distance(vertex, segment)
            if distance < min_distance:
                min_distance = distance
                closest_point = vertex

        return closest_point

    def cut_free_space_with_point_cloud(
        self, near_side, far_side, pivot_point, force=False
    ):
        closest_point_between_sides, min_distance = far_side[0], (
            far_side[0] - pivot_point
        ).dot(far_side[0] - pivot_point)
        for vertex in self.vertices:
            if (
                not Basic2DGeometry.point_in_parallel_lines(vertex, near_side, far_side)
                and not force
            ):
                continue
            distance = (pivot_point - vertex).dot(pivot_point - vertex)
            if distance < min_distance:
                min_distance = distance
                closest_point_between_sides = vertex

        if min_distance > 1e9 or min_distance == np.inf:
            return near_side, far_side
        # print('!closest_point_between_sides!', closest_point_between_sides)
        closest_line = np.array(
            [
                closest_point_between_sides,
                near_side[1] - near_side[0] + closest_point_between_sides,
            ]
        )
        # print('!closest_line!', closest_line)
        new_segment = Basic2DGeometry.point_to_line(
            near_side[0], closest_line
        ), Basic2DGeometry.point_to_line(near_side[1], closest_line)
        # print('!new_segment!\n', new_segment,'\n\n')
        return near_side, new_segment

    def cut_free_space_with_convex(self, near_side, far_side, force=False):

        # self.heading = (near_side[1] - near_side[0]) / np.linalg.norm(near_side[1] - near_side[0])
        # headed_bbox = self.get_headed_bbox()
        vertex_side = [
            Basic2DGeometry.point_side_of_line(vertex, near_side)
            for vertex in self.vertices[self.convex_hull.vertices]
        ]
        if (1 in vertex_side) and (-1 in vertex_side):
            if force:
                return near_side, near_side
            return near_side, far_side

        closest_point_near_side = self.get_closest_point_to_line(
            near_side, default_point=far_side[0]
        )
        if not Basic2DGeometry.point_in_parallel_lines(
            closest_point_near_side, near_side, far_side
        ):
            if force:
                return near_side, near_side
            return near_side, far_side
        closest_line = [
            closest_point_near_side,
            near_side[1] - near_side[0] + closest_point_near_side,
        ]
        # print('closest_line', closest_line)
        new_segment = Basic2DGeometry.point_to_line(
            near_side[0], closest_line
        ), Basic2DGeometry.point_to_line(near_side[1], closest_line)
        # if max(abs(new_segment)) > 100:
        #     print('new_segment', new_segment)
        assert (
            abs((near_side[1] - near_side[0]).dot(new_segment[0] - near_side[0]))
            <= 1e-6
        )
        t = new_segment[1] - new_segment[0] - (near_side[1] - near_side[0])
        assert t.dot(t) <= 1e-6

        return near_side, new_segment
        pass

    def intersect_with_another_convex(self, other):
        polygon1 = Polygon(self.vertices[self.convex_hull.vertices])
        polygon2 = Polygon(other.vertices[other.convex_hull.vertices])
        intersection = polygon1.intersection(polygon2)
        intersection_vertices = []
        if isinstance(intersection, Polygon):
            intersection_vertices = list(intersection.exterior.coords)
        elif (
            isinstance(intersection, LineString)
            or isinstance(intersection, Point)
            or isinstance(intersection, LinearRing)
        ):
            intersection_vertices = list(intersection.coords)

        if intersection is None or len(intersection_vertices) < 3:
            return None
        intersection_convex_hull = ConvexHullProcessor_2d(
            vertices=intersection_vertices
        )
        if intersection.area < 1e-4:
            return None
        return intersection_convex_hull

    def intersect_area_with_another_convex(self, other):
        polygon1 = Polygon(self.vertices[self.convex_hull.vertices])
        polygon2 = Polygon(other.vertices[other.convex_hull.vertices])
        intersection = polygon1.intersection(polygon2)
        return intersection.area

    def is_intersected_with_rectangle(self, rectangle):
        vertices = self.vertices
        for vertex in vertices:
            if Basic2DGeometry.is_inside_rectangle(vertex, rectangle):
                return True
        return False


class Basic2DGeometry:

    def __init__(self):
        pass

    @staticmethod
    def rotate_points(points, angle):
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.dot(points, R)

    @staticmethod
    def point_to_segment_distance(point, segment):
        start, end = segment[0], segment[1]
        if np.allclose(start, end):
            return np.linalg.norm(point - start)

        direction = end - start
        t = np.dot(point - start, direction) / np.dot(direction, direction)
        if t < 0:
            return np.linalg.norm(point - start)
        elif t > 1:
            return np.linalg.norm(point - end)
        else:
            projection = start + t * direction
            return np.linalg.norm(point - projection)

    @staticmethod
    def is_inside_rectangle(point, rect):

        signs = [
            (rect[i][0] - point[0]) * (rect[(i + 1) % 4][1] - point[1])
            - (rect[(i + 1) % 4][0] - point[0]) * (rect[i][1] - point[1])
            for i in range(4)
        ]
        return all(sign > 1e-6 for sign in signs) or all(sign < -1e-6 for sign in signs)

    @staticmethod
    def is_on_segment(point, segment):
        start, end = segment[0], segment[1]
        if np.allclose(start, end):
            return np.allclose(point, start)

        direction = end - start
        t = np.dot(point - start, direction) / np.dot(direction, direction)
        return 0 <= t <= 1

    @staticmethod
    def intersection_of_line(line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        return np.array([px, py])

    @staticmethod
    def intersection_of_parallel_rectangle(rect1, rect2):
        intersection_points = []
        for point in rect1:
            if Basic2DGeometry.is_inside_rectangle(point, rect2):
                intersection_points.append(point)
        for point in rect2:
            if Basic2DGeometry.is_inside_rectangle(point, rect1):
                intersection_points.append(point)

        for i in range(4):
            for j in range(4):
                intersection = Basic2DGeometry.intersection_of_line(
                    [rect1[i], rect1[(i + 1) % 4]], [rect2[j], rect2[(j + 1) % 4]]
                )
                if intersection is not None:
                    intersection_points.append(intersection)

        if len(intersection_points) < 4:
            return None

        intersection_polygon = ConvexHullProcessor_2d(vertices=intersection_points)
        return intersection_polygon

    @staticmethod
    def point_side_of_line(point, line):
        return np.sign(np.cross(line[1] - line[0], point - line[0]))

    @staticmethod
    def closest_point_on_line(point, a, b, c):
        x = (b * (b * point[0] - a * point[1]) - a * c) / (a**2 + b**2)
        y = (a * (-b * point[0] + a * point[1]) - b * c) / (a**2 + b**2)
        return np.array([x, y])

    @staticmethod
    def point_to_line_distance(point, line):
        return np.linalg.norm(
            np.cross(line[1] - line[0], line[0] - point)
        ) / np.linalg.norm(line[1] - line[0])

    @staticmethod
    def line2abc(line):
        start, end = line[0], line[1]
        a = end[1] - start[1]
        b = start[0] - end[0]
        c = -a * start[0] - b * start[1]
        return a, b, c

    @staticmethod
    def point_to_line(point, line):
        a, b, c = Basic2DGeometry.line2abc(line)
        return Basic2DGeometry.closest_point_on_line(point, a, b, c)

    @staticmethod
    def point_in_polygon(point, polygon):
        n = len(polygon)
        polygon.append(polygon[0])
        signs = [
            (polygon[i][0] - point[0]) * (polygon[i + 1][1] - point[1])
            - (polygon[i + 1][0] - point[0]) * (polygon[i][1] - point[1])
            for i in range(n)
        ]
        return all(sign > 0 for sign in signs) or all(sign < 0 for sign in signs)

    @staticmethod
    def point_in_parallel_lines(point, line1, line2):
        direction1 = (line1[1] - line1[0]) / np.linalg.norm(line1[1] - line1[0])
        direction2 = (line2[1] - line2[0]) / np.linalg.norm(line2[1] - line2[0])

        start1, start2 = line1[0], line2[0]

        perpendicular_direction1 = np.array([-direction1[1], direction1[0]])
        perpendicular_direction2 = np.array([-direction2[1], direction2[0]])

        distance_to_line1 = np.dot(point - start1, perpendicular_direction1)
        distance_to_line2 = np.dot(point - start2, perpendicular_direction2)

        return (distance_to_line1 >= -1e-4 and distance_to_line2 <= 1e-4) or (
            distance_to_line1 <= 1e-4 and distance_to_line2 >= -1e-4
        )

    @staticmethod
    def cal_vector_cos(vector1, vector2):
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )


class Basic3Dgeometry:

    def __init__(self):
        pass

    @staticmethod
    def point_to_line_distance(point, line):
        return np.linalg.norm(
            np.cross(line[1] - line[0], line[0] - point)
        ) / np.linalg.norm(line[1] - line[0])


def quaternion_between_vectors(v1, v2):
    """
    计算从向量 v1 旋转到向量 v2 的旋转四元数。

    参数:
    - v1: 第一个向量 [x1, y1, z1]
    - v2: 第二个向量 [x2, y2, z2]

    返回:
    - 旋转四元数 [x, y, z, w]
    """
    # 归一化向量
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # 计算旋转轴
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # 如果向量平行或反平行，特殊处理
    if axis_norm == 0:
        if np.dot(v1, v2) > 0:
            return np.array([0, 0, 0, 1])  # 平行向量，无需旋转
        else:
            # 反平行向量，选择任意垂直向量作为旋转轴
            orthogonal_vector = (
                np.array([1, 0, 0]) if abs(v1[0]) < abs(v1[1]) else np.array([0, 1, 0])
            )
            axis = np.cross(v1, orthogonal_vector)
            axis = axis / np.linalg.norm(axis)
            return np.concatenate((axis, [0]))

    axis = axis / axis_norm

    # 计算旋转角度
    angle = np.arccos(np.dot(v1, v2))

    # 计算旋转四元数
    quaternion = R.from_rotvec(angle * axis).as_quat()

    return quaternion


def quaternion_from_vector(vector, angle):
    """
    计算给定向量的旋转四元数。

    参数:
    - vector: 旋转轴向量 [x, y, z]
    - angle: 旋转角度（弧度）

    返回:
    - 旋转四元数 [x, y, z, w]
    """
    # 归一化向量
    vector = vector / np.linalg.norm(vector)

    # 计算旋转四元数
    quaternion = R.from_rotvec(angle * vector).as_quat()

    return quaternion


def vector_to_rpy(vector):

    # 归一化向量
    vector = vector / np.linalg.norm(vector)

    # 计算旋转矩阵
    z_axis = vector
    x_axis = np.array([1, 0, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # 计算 RPY 角
    r = R.from_matrix(rotation_matrix)
    rpy = r.as_euler("xyz", degrees=True)

    return rpy
