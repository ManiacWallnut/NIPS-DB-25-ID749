import numpy as np
from bisect import bisect_left, bisect_right


class PointSumQuery2D:
    def __init__(self, points, values):

        import time

        start = time.time()

        x_coords = sorted(set([point[0] for point in points]))
        y_coords = sorted(set([point[1] for point in points]))

        self.x_map = {x: i for i, x in enumerate(x_coords)}
        self.y_map = {y: i for i, y in enumerate(y_coords)}
        self.x_vals = x_coords
        self.y_vals = y_coords

        n, m = len(x_coords), len(y_coords)

        self.sum = np.zeros((n + 1, m + 1))
        grid = np.zeros((n, m))

        for (x, y), val in zip(points, values):
            grid[self.x_map[x], self.y_map[y]] += val

        grid_cumsum = np.cumsum(np.cumsum(grid, axis=0), axis=1)
        self.sum[1:, 1:] = grid_cumsum
        import glog

        glog.info(
            f"Time taken to build the 2D point sum query processor:  {time.time()-start}"
        )

    def query_inclusive(self, x, y):

        xi = bisect_right(self.x_vals, x)
        yi = bisect_right(self.y_vals, y)

        return self.sum[xi][yi]

    def query_exclusive(self, x, y):

        xi = bisect_left(self.x_vals, x)
        yi = bisect_left(self.y_vals, y)

        return self.sum[xi][yi]

    def query_rect(self, x1, y1, x2, y2):

        return (
            self.query_inclusive(x2, y2)
            - self.query_exclusive(x1, y2)
            - self.query_exclusive(x2, y1)
            + self.query_inclusive(x1, y1)
        )


class PointSumQuery2D_SegmentTree:
    EPS = 1e-3

    class Node:
        def __init__(self, x1, y1, x2, y2, sum=0):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.sum = sum
            self.nw_child = None
            self.ne_child = None
            self.sw_child = None
            self.se_child = None

        def split(self):
            x_mid = (self.x1 + self.x2) * 0.5
            y_mid = (self.y1 + self.y2) * 0.5

            self.nw_child = PointSumQuery2D_SegmentTree.Node(
                self.x1, self.y1, x_mid, y_mid
            )
            self.ne_child = PointSumQuery2D_SegmentTree.Node(
                x_mid, self.y1, self.x2, y_mid
            )
            self.sw_child = PointSumQuery2D_SegmentTree.Node(
                self.x1, y_mid, x_mid, self.y2
            )
            self.se_child = PointSumQuery2D_SegmentTree.Node(
                x_mid, y_mid, self.x2, self.y2
            )

        def update_sum(self):
            self.sum = 0
            if self.nw_child is not None:
                self.sum += self.nw_child.sum
            if self.ne_child is not None:
                self.sum += self.ne_child.sum
            if self.sw_child is not None:
                self.sum += self.sw_child.sum
            if self.se_child is not None:
                self.sum += self.se_child.sum
            return self.sum

    def __init__(self):
        self.root = None

    def _add_rectangle(self, node, x1, y1, x2, y2, val):
        if x1 >= node.x2 or x2 <= node.x1 or y1 >= node.y2 or y2 <= node.y1:
            return
        if (x1 <= node.x1 and x2 >= node.x2 and y1 <= node.y1 and y2 >= node.y2) or (
            node.x2 - node.x1
        ) * (node.y2 - node.y1) < self.EPS * self.EPS:
            node.sum += val * (x2 - x1) * (y2 - y1)
            return
        if (
            node.nw_child is None
            or node.ne_child is None
            or node.sw_child is None
            or node.se_child is None
        ):
            node.split()
        self._add_rectangle(node.nw_child, x1, y1, x2, y2, val)
        self._add_rectangle(node.ne_child, x1, y1, x2, y2, val)
        self._add_rectangle(node.sw_child, x1, y1, x2, y2, val)
        self._add_rectangle(node.se_child, x1, y1, x2, y2, val)
        node.update_sum()

    def _query_rectangle(self, node, x1, y1, x2, y2):
        if x1 >= node.x2 or x2 <= node.x1 or y1 >= node.y2 or y2 <= node.y1:
            return 0
        if (
            (x1 <= node.x1 and x2 >= node.x2 and y1 <= node.y1 and y2 >= node.y2)
            or (node.x2 - node.x1) < self.EPS
            or (node.y2 - node.y1) < self.EPS
        ):
            return node.sum
        return (
            self._query_rectangle(node.nw_child, x1, y1, x2, y2)
            + self._query_rectangle(node.ne_child, x1, y1, x2, y2)
            + self._query_rectangle(node.sw_child, x1, y1, x2, y2)
            + self._query_rectangle(node.se_child, x1, y1, x2, y2)
        )

    def add_rectangle(self, x1, y1, x2, y2, val):
        if self.root is None:
            self.root = self.Node(x1, y1, x2, y2, val)
        else:
            self._add_rectangle(self.root, x1, y1, x2, y2, val)

    def query_rect(self, x1, y1, x2, y2):
        if self.root is None:
            return 0
        return self._query_rectangle(self.root, x1, y1, x2, y2)
