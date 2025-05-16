import trimesh
import numpy as np
from scipy.spatial import cKDTree
from typing import List
from shapely.geometry import Polygon, LinearRing, LineString, Point
from .affordable_platform import AffordablePlatform
from .polygon_processor import PolygonProcessor
from .convex_hull_processor import ConvexHullProcessor_2d, Basic2DGeometry


def reindex(vertex_indices, old_faces):
    new_faces = []
    for face in old_faces:
        new_face = [
            np.where(vertex_indices == face[i])[0][0] for i in range(face.shape[0])
        ]
        new_faces.append(new_face)
    return new_faces


def cal_vector_cos(vector1, vector2):
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


class MeshProcessor:
    # minimal size of the platform
    min_size = 0.0025
    relative_size_ratio = 0.25

    # mainly used to process the mesh
    def __init__(self, mesh=None, name=""):
        # mesh is a trimesh object
        self.mesh = mesh
        # name of the mesh
        self.name = name
        # list of platforms that are affordable. Each platform consists of a list of faces.
        self.affordable_platforms = []

    def __repr__(self):
        return f"MeshProcessor(name={self.name})"

    def from_faces_and_vertices(faces, vertices):
        return MeshProcessor(trimesh.Trimesh(vertices=vertices, faces=faces))

    def get_bounding_box(self):
        return self.mesh.bounds

    def calculate_normal_vector(self):
        # 计算法向量
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        res = []
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            face_normal /= np.linalg.norm(face_normal)
            res.append(face_normal)
        return res

    @staticmethod
    def repair_mesh(mesh):
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        return mesh

    def repair_mesh_instance(self):

        trimesh.repair.fix_inversion(self.mesh)
        trimesh.repair.fix_winding(self.mesh)
        trimesh.repair.fix_normals(self.mesh)
        trimesh.repair.fill_holes(self.mesh)
        return self

    def cal_orientation(self):

        vertices_2d = self.mesh.vertices[:, :2]
        vertices_2d += np.random.rand(*vertices_2d.shape) * 1e-6

        bounds_2d = trimesh.bounds.oriented_bounds_2D(vertices_2d)
        x_min = bounds_2d[1][0] * (-0.5)
        x_max = bounds_2d[1][0] * 0.5
        y_min = bounds_2d[1][1] * (-0.5)
        y_max = bounds_2d[1][1] * 0.5

        bounding_points = np.array(
            [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
        )
        bounding_points -= bounds_2d[0][2, :2]

        rotation_matrix_inv = np.linalg.inv(bounds_2d[0][:2, :2])
        bounding_points = bounding_points @ rotation_matrix_inv.T

        return bounding_points, (rotation_matrix_inv[0, 0], rotation_matrix_inv[1, 0])

    def cal_convex_hull_2d(self):
        vertices_2d = self.mesh.vertices[:, :2]
        convex_hull = ConvexHullProcessor_2d(vertices_2d)
        return convex_hull

    def mesh_after_merge_close_vertices(self, tol=1e-6):
        # use CKDTree to deal with mesh with coincident vertices
        tree = cKDTree(self.mesh.vertices)
        unique_indices = tree.query_ball_tree(tree, tol)

        # create a map to merge vertices
        merge_map = {}
        for group in unique_indices:
            representative = group[0]
            for idx in group:
                merge_map[idx] = representative

        # create new vertices and faces
        new_vertices = []
        new_faces = []
        vertex_map = {}
        for _, new_idx in merge_map.items():
            if new_idx not in vertex_map:
                vertex_map[new_idx] = len(new_vertices)
                new_vertices.append(self.mesh.vertices[new_idx])

        for face in self.mesh.faces:
            new_face = [vertex_map[merge_map[idx]] for idx in face]
            new_faces.append(new_face)

        new_vertices = np.array(new_vertices)
        new_faces = np.array(new_faces)

        self.mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        return self

    def find_vertex_and_faces(mesh, target_vertex):
        matching_indices = np.where(
            np.all(np.isclose(mesh.vertices, target_vertex, atol=1e-2), axis=1)
        )[0]

        if len(matching_indices) == 0:
            print(f"No vertex found at position {target_vertex}")
            return

        for idx in matching_indices:
            print(f"Vertex found at index {idx}: {mesh.vertices[idx]}")
            faces_with_vertex = np.where(np.any(mesh.faces == idx, axis=1))[0]
            for face_idx in faces_with_vertex:
                print(f"Face {face_idx}: {mesh.faces[face_idx]}")

    @staticmethod
    def create_cuboid_from_vertices(vertices):
        # 确保顶点顺序符合右手法则
        vertices = np.array(vertices)

        # 计算底面和顶面的法向量
        bottom_normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[1])
        top_normal = np.cross(vertices[5] - vertices[4], vertices[6] - vertices[5])

        # 如果底面法向量和顶面法向量方向相反，则交换顶面顶点顺序
        if np.dot(bottom_normal, top_normal) < 0:
            vertices[4], vertices[5], vertices[6], vertices[7] = (
                vertices[4],
                vertices[7],
                vertices[6],
                vertices[5],
            )

        # 定义长方体的面，确保顶点顺序符合右手法则
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Bottom face
                [4, 6, 5],
                [4, 7, 6],  # Top face
                [0, 5, 1],
                [0, 4, 5],  # Front face
                [1, 6, 2],
                [1, 5, 6],  # Right face
                [2, 7, 3],
                [2, 6, 7],  # Back face
                [3, 4, 0],
                [3, 7, 4],  # Left face
            ]
        )

        # 创建长方体网格
        cuboid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if cuboid_mesh.volume < 0:
            faces = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],  # Bottom face
                    [4, 5, 6],
                    [4, 6, 7],  # Top face
                    [0, 1, 5],
                    [0, 5, 4],  # Front face
                    [1, 2, 6],
                    [1, 6, 5],  # Right face
                    [2, 3, 7],
                    [2, 7, 6],  # Back face
                    [3, 0, 4],
                    [3, 4, 7],  # Left face
                ]
            )
            cuboid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return MeshProcessor(cuboid_mesh)

    @staticmethod
    def create_cuboid_from_bbox(bbox_min, bbox_max):
        # Define the 8 vertices of the cuboid
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

        # Define the 12 faces of the cuboid
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Bottom face
                [4, 5, 6],
                [4, 6, 7],  # Top face
                [0, 1, 5],
                [0, 5, 4],  # Front face
                [1, 2, 6],
                [1, 6, 5],  # Right face
                [2, 3, 7],
                [2, 7, 6],  # Back face
                [3, 0, 4],
                [3, 4, 7],  # Left face
            ]
        )

        # Create the cuboid mesh
        cuboid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return MeshProcessor(cuboid_mesh)

    def intersection_with_cuboid(self, cuboid_mesh):

        cuboid_mesh.repair_mesh_instance()
        self.repair_mesh_instance()
        if self.mesh.is_volume == False:
            #  print('mesh is not volume')
            return None
        if cuboid_mesh.mesh.is_volume == False:

            #  print('cuboid is not volume')
            return None
        # print('bbox',bbox_min, bbox_max, self.mesh.bounds)
        intersection_mesh = trimesh.boolean.boolean_manifold(
            [cuboid_mesh.mesh, self.mesh], operation="intersection"
        )
        if intersection_mesh.vertices.shape[0] < 4:
            if intersection_mesh.vertices.shape[0] == 0:
                pass
                # print('intersection is empty')
            else:
                print(intersection_mesh.vertices, "intersection is not a volume!")
                if intersection_mesh.vertices.shape[0] == 3:
                    intersection_mesh = (
                        MeshProcessor.create_thin_cylinder_from_three_points(
                            intersection_mesh.vertices
                        )
                    )
                else:
                    return None

            return None
        intersection_mesh = MeshProcessor.repair_mesh(intersection_mesh)
        return MeshProcessor(intersection_mesh)

    @staticmethod
    def create_thin_cylinder_from_three_points(points, thickness=1e-2):
        # 计算三个点的中心点
        center = np.mean(points, axis=0)

        # 计算法向量
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        # 创建两个平行的三角形
        offset = normal * thickness / 2
        vertices = np.vstack([points + offset, points - offset])

        # 定义柱体的面
        faces = np.array(
            [
                [0, 1, 2],  # 上表面三角形
                [3, 4, 5],  # 下表面三角形
                [0, 1, 4],
                [0, 4, 3],  # 侧面四边形分成两个三角形
                [1, 2, 5],
                [1, 5, 4],
                [2, 0, 3],
                [2, 3, 5],
            ]
        )

        # 创建 trimesh 对象
        thin_cylinder = trimesh.Trimesh(vertices=vertices, faces=faces)

        return thin_cylinder

    @staticmethod
    def create_cylinder_from_polygon(polygon, z_bottom, z_top):
        meshes = []
        polygons = polygon.polygons
        for polygon in polygons:
            if isinstance(polygon, Polygon):
                trimesh_polygon = trimesh.creation.extrude_polygon(
                    polygon, height=z_top - z_bottom
                )
                trimesh_polygon.apply_translation([0, 0, z_bottom])
                meshes.append(trimesh_polygon)
            elif isinstance(polygon, LineString):
                coords = np.array(polygon.coords)
                for i in range(len(coords) - 1):
                    start = coords[i]
                    end = coords[i + 1]
                    length = np.linalg.norm(end - start)
                    direction = (end - start) / length
                    width = 0.1
                    height = z_top - z_bottom
                    rect = trimesh.creation.box(extents=[length, width, height])
                    # create a transformation matrix to move the rectangle to the right position
                    transform = np.eye(4)
                    transform[:2, 3] = (start + end) / 2
                    rect.apply_transform(transform)
                    rect.apply_translation([0, 0, z_bottom])
                    meshes.append(rect)
            elif isinstance(polygon, Point):
                # Turn the point into a sphere
                sphere = trimesh.creation.icosphere(radius=0.1)
                sphere.apply_translation([polygon.x, polygon.y, (z_bottom + z_top) / 2])
                meshes.append(sphere)
            else:
                print(f"Unsupported geometry type: {type(polygon)}")
                continue
        # 合并所有网格
        combined_mesh = trimesh.util.concatenate(meshes)
        # Rotate the mesh to swap the z and y coordinates
        # rotation_matrix = trimesh.transformations.rotation_matrix(3*np.pi / 2, [1, 0, 0])

        # combined_mesh.apply_transform(rotation_matrix)

        combined_mesh = MeshProcessor(combined_mesh).repair_mesh_instance()

        return combined_mesh

    def get_raw_affordable_platforms(
        self,
        base_thres=np.cos(np.deg2rad(20)),
        adj_thres=np.deg2rad(20),
        abs_thres=np.cos(np.deg2rad(40)),
    ):
        separate_face_list = []
        geometry = self.mesh

        normal_vector = self.calculate_normal_vector()
        vertical_vector = np.array([0, 0, 1])
        n_face = geometry.faces.shape[0]
        adjacency_faces = geometry.face_adjacency
        adjacency_angles = geometry.face_adjacency_angles
        adjacency = [set() for _ in range(n_face)]

        for i in range(len(adjacency_faces)):
            face1, face2 = adjacency_faces[i]
            angle = adjacency_angles[i]
            adjacency[face1].add((face2, angle))
            adjacency[face2].add((face1, angle))

        # calculate the cos_value between normal_vector and vertical_vector
        cos_values = [
            cal_vector_cos(normal_vector[i], vertical_vector) for i in range(n_face)
        ]
        affordable_face_indices = {
            i for i in range(n_face) if cos_values[i] > base_thres
        }

        bel = [i for i in range(n_face)]

        def getf(x):
            return x if bel[x] == x else getf(bel[x])

        while True:
            new_faces = set()
            for face_indice in affordable_face_indices:
                for adj_face_indice, angle in adjacency[face_indice]:
                    if (
                        cos_values[adj_face_indice] > abs_thres
                        and abs(angle) < adj_thres
                    ):
                        bel[getf(adj_face_indice)] = getf(face_indice)
                        if (
                            adj_face_indice not in affordable_face_indices
                            and adj_face_indice not in new_faces
                        ):
                            new_faces.add(adj_face_indice)
            if len(new_faces) == 0:
                break
            affordable_face_indices.update(new_faces)

        separate_faces = {}
        for face in affordable_face_indices:
            if getf(face) not in separate_faces:
                separate_faces[getf(face)] = []
            separate_faces[getf(face)].append(face)

        separate_face_id = 0
        for v in separate_faces.values():
            face_indices = v
            faces = [geometry.faces[i] for i in face_indices]
            vertices = np.unique(faces)
            new_faces = reindex(vertices, faces)
            new_vertices = geometry.vertices[vertices]
            tmp_affordable_platform = AffordablePlatform(
                vertices=new_vertices,
                faces=new_faces,
                name=self.name + "_" + str(separate_face_id),
            )
            if tmp_affordable_platform.get_area() > self.min_size:
                separate_face_id += 1
                separate_face_list.append(tmp_affordable_platform)

        self.affordable_platforms = separate_face_list

        return separate_face_list

    # check if other_platform is overlapped with the current platform.
    # if true, calculate the intersected volume and determine if there are space between these faces.
    # if there are no space, create a derivative object for this platform.
    # if there are space, update the available height as min(available_height, self.bottom - other.top)

    def calculate_affordable_platforms(self, raw_platform=True, top_area=0):

        new_derivatives = []

        for i in range(len(self.affordable_platforms)):
            self.affordable_platforms[i].is_top_platform = True

        self.repair_mesh_instance()
        for i in range(len(self.affordable_platforms)):
            top_polygon = self.affordable_platforms[i].get_polygon()
            top_height = self.affordable_platforms[i].get_height()[1]
            for j in range(len(self.affordable_platforms)):
                if i == j:
                    continue
                other_platform = self.affordable_platforms[j]
                other_polygon = other_platform.get_polygon()
                other_height = other_platform.get_height()[0]
                if top_height - other_height < 1e-6:
                    continue
                intersected_polygon = PolygonProcessor.intersect_two_polygons(
                    top_polygon, other_polygon
                )
                intersected_pieces = intersected_polygon.get_len()
                intersected_area = PolygonProcessor.intersect_two_polygons(
                    top_polygon, other_polygon
                ).get_area()

                top_polygon_area = PolygonProcessor.intersect_two_polygons(
                    other_polygon, other_polygon
                ).get_area()

                if intersected_pieces == 0 or intersected_area < 1e-6:
                    continue

                potential_occupied_space = MeshProcessor.create_cylinder_from_polygon(
                    intersected_polygon, other_height, top_height
                ).repair_mesh_instance()

                self.affordable_platforms[j].is_top_platform = False

                if potential_occupied_space.mesh.is_volume == False:
                    potential_occupied_space.mesh = (
                        potential_occupied_space.mesh.convex_hull
                    )

                if (
                    potential_occupied_space.mesh.is_volume == False
                    or self.mesh.is_volume == False
                ):
                    continue
                    # potential_occupied_space.mesh = potential_occupied_space.mesh.convex_hull
                intersection_mesh = trimesh.boolean.boolean_manifold(
                    [potential_occupied_space.mesh, self.mesh], operation="intersection"
                )

                intersection_mesh = MeshProcessor.repair_mesh(intersection_mesh)
                intersection_volume = intersection_mesh.volume
                if intersection_volume < 1e-5:
                    continue
                # print(intersection_mesh.bounds, potential_occupied_space.mesh.bounds, self.mesh.bounds )
                intersection_on_surface = (
                    top_height
                    - other_height
                    - (
                        potential_occupied_space.mesh.bounds[1][2]
                        - intersection_mesh.bounds[0][2]
                    )
                    < 2e-2
                )

                if intersection_on_surface and not raw_platform:
                    derivative = object(geometries=[intersection_mesh.convex_hull])
                    self.affordable_platforms[j].derivative.append(derivative)
                    new_derivatives.append(derivative)
                else:
                    if intersection_volume > 0.25 * top_polygon_area * (
                        top_height - other_height
                    ):
                        self.affordable_platforms[j].available_height = min(
                            self.affordable_platforms[j].available_height,
                            float(intersection_mesh.bounds[0][2] - other_height),
                        )
                    else:
                        self.affordable_platforms[j].available_height = min(
                            self.affordable_platforms[j].available_height,
                            top_height - other_height,
                        )
        self.clear_small_platforms()
        return new_derivatives

    def clear_small_platforms(self):
        if len(self.affordable_platforms) == 0:
            return self
        max_area = np.max(
            [platform.get_area() for platform in self.affordable_platforms]
        )
        self.affordable_platforms = [
            platform
            for platform in self.affordable_platforms
            if platform.get_area() > max_area * self.relative_size_ratio
        ]
        return self
