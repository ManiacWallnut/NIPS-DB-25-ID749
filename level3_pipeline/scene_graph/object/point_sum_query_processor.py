import numpy as np
from bisect import bisect_left, bisect_right


class PointSumQuery2D:
    def __init__(self, points, values):

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

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                self.sum[i][j] = (
                    grid[i - 1][j - 1]
                    + self.sum[i - 1][j]
                    + self.sum[i][j - 1]
                    - self.sum[i - 1][j - 1]
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
            self.query_exclusive(x2, y2)
            - self.query_inclusive(x1, y2)
            - self.query_inclusive(x2, y1)
            + self.query_exclusive(x1, y1)
        )
