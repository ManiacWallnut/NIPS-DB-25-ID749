from shapely.geometry import Polygon, LinearRing, LineString, Point
import matplotlib.pyplot as plt
from typing import List


class PolygonProcessor:
    def __init__(self, polygons: List[Polygon]):
        self.polygons = polygons

    def from_trimesh_geometry(geometry):
        pass

    def plot_polygons(self, title="Polygons"):
        polygons = self.polygons
        fig, ax = plt.subplots()
        for geom in polygons:
            if geom.is_empty:
                continue
            if isinstance(geom, Polygon):
                exterior = LinearRing(geom.exterior.coords)
                x, y = exterior.xy
                ax.plot(
                    x,
                    y,
                    color="blue",
                    alpha=0.7,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.fill(x, y, color="skyblue", alpha=0.4)
                # inner cycle
                for interior in geom.interiors:
                    ring = LinearRing(interior.coords)
                    x, y = ring.xy
                    ax.plot(
                        x,
                        y,
                        color="red",
                        alpha=0.7,
                        linewidth=2,
                        solid_capstyle="round",
                        zorder=2,
                    )
                    ax.fill(x, y, color="lightcoral", alpha=0.4)
            elif isinstance(geom, LineString):
                x, y = geom.xy
                ax.plot(
                    x,
                    y,
                    color="green",
                    alpha=0.7,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=2,
                )
            elif isinstance(geom, Point):
                x, y = geom.xy
                ax.plot(x, y, "o", color="purple", alpha=0.7, markersize=5, zorder=2)
            else:
                print(f"Unsupported geometry type: {type(geom)}")
        ax.set_title(title)
        plt.show()

    @staticmethod
    def mesh_to_polygon(mesh):
        vertices = mesh.vertices[:, :2]
        faces = mesh.faces
        polygons = []
        for face in faces:
            polygon = Polygon(vertices[face])
            polygons.append(polygon)
        return polygons

    @staticmethod
    def get_convex_hull(vertices):
        return Polygon(vertices).convex_hull

    @staticmethod
    def vertices_and_face_to_polygon(vertices, faces):
        vertices = vertices[:, :2]
        polygons = []
        for face in faces:
            polygon = Polygon(vertices[face])
            polygons.append(polygon)
        return polygons

    @staticmethod
    def intersect_two_polygons(polygon1, polygon2):
        intersection = []
        for poly1 in polygon1:
            for poly2 in polygon2:
                inter = poly1.intersection(poly2)
                if not inter.is_empty:
                    intersection.append(inter)
        return PolygonProcessor(intersection)

    def intersect_polygons(self, polygons):
        polygons1 = self.polygons
        polygons2 = polygons
        return PolygonProcessor.intersect_two_polygons(polygons1, polygons2)

    def get_len(self):
        return len(self.polygons)

    def get_area(self):
        return sum([poly.area for poly in self.polygons])
