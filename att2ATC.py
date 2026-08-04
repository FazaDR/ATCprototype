from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set
import numpy as np
from pathlib import Path
import open3d as o3d

class TopologicalMap:
    """
    Topological map used by ATC-DT.

    Corresponds to

        G = (V, {h_i}, E)

    where

        V          : node IDs
        positions  : h_pos
        normals    : h_nor (future)
        traversability : h_tra (future)
        graph      : E_pos
    """

    def __init__(self):

        # -----------------------------
        # Node set V
        # -----------------------------

        self.node_count = 0

        # -----------------------------
        # Reference vectors
        # -----------------------------

        # h_pos
        self.positions = np.empty((0, 3), dtype=np.float32)

        # h_nor
        # (not used yet)
        self.normals = np.empty((0, 3), dtype=np.float32)

        # h_tra
        # (not used yet)
        self.traversability = np.empty((0,), dtype=np.int8)

        # -----------------------------
        # Edge set E_pos
        # graph[a][b] = edge age
        # -----------------------------

        self.graph: Dict[int, Dict[int, int]] = defaultdict(dict)

    def edges(self):

        visited = set()

        edges = []

        for a in self.graph:

            for b in self.graph[a]:

                if (b, a) in visited:
                    continue

                visited.add((a, b))
                edges.append((a, b))

        return edges

    # ==========================================================
    # Node Operations
    # ==========================================================

    def add_node(self, position: np.ndarray) -> int:
        """
        Add a new node to the topological map.
        Returns the newly assigned node ID.
        """

        position = np.asarray(position, dtype=np.float32)

        self.positions = np.vstack([
            self.positions,
            position.reshape(1, 3)
        ])

        # Placeholder values for future attributes
        self.normals = np.vstack([
            self.normals,
            np.zeros((1, 3), dtype=np.float32)
        ])

        self.traversability = np.append(
            self.traversability,
            0
        )

        node_id = self.node_count
        self.node_count += 1

        return node_id

    # ==========================================================
    # Edge Operations
    # ==========================================================

    def has_edge(self, a: int, b: int) -> bool:

        return b in self.graph[a]

    def add_edge(self, a: int, b: int):

        self.graph[a][b] = 0
        self.graph[b][a] = 0

    def remove_edge(self, a: int, b: int):

        self.graph[a].pop(b, None)
        self.graph[b].pop(a, None)

    def neighbors(self, node_id: int):

        return self.graph[node_id].keys()

    # ==========================================================
    # Utilities
    # ==========================================================

    def edge_ages(self):

        ages = []

        visited = set()

        for a in self.graph:
            for b, age in self.graph[a].items():

                if (b, a) in visited:
                    continue

                visited.add((a, b))
                ages.append(age)

        return np.asarray(ages, dtype=np.float32)

    def num_edges(self):

        count = 0

        visited = set()

        for a in self.graph:
            for b in self.graph[a]:

                if (b, a) in visited:
                    continue

                visited.add((a, b))
                count += 1

        return count




class ATCDT:

    def __init__(
        self,
        vigilance=0.5,
        lambda_points=4000
    ):

        self.vigilance = vigilance
        self.lambda_points = lambda_points

        # Γdel
        self.deleted_edge_ages = []

        # G = (V, h_pos, E)
        self.map = TopologicalMap()

        # m_i in Eq. (6)
        self.winner_count = np.empty((0,), dtype=np.int32)

    # def update_normal_map(self):
    #     pass

    # def update_traversability_map(self):
    #     pass

    def sample_points(self, points):

        n = len(points)

        if n <= self.lambda_points:
            return points

        idx = np.random.choice(
            n,
            self.lambda_points,
            replace=False
        )

        return points[idx]

    def add_node(self, point):

        node_id = self.map.add_node(point)

        self.winner_count = np.concatenate(
            (self.winner_count, np.array([0], dtype=np.int32))
        )

        return node_id

    def winner_search(self, point):

        if self.map.node_count == 0:
            return None, None, np.inf, np.inf

        dist = np.linalg.norm(
            self.map.positions - point,
            axis=1
        )

        if self.map.node_count == 1:

            d = np.linalg.norm(self.map.positions[0] - point)

            return 0, None, d, np.inf

        order = np.argpartition(dist, 1)

        s1 = order[0]
        s2 = order[1]

        if dist[s2] < dist[s1]:
            s1, s2 = s2, s1

        d1 = dist[s1]
        d2 = dist[s2]

        return s1, s2, d1, d2


    def update_existing_node(
        self,
        point,
        winner
    ):

        self.winner_count[winner] += 1

        lr = 1.0 / (10 * self.winner_count[winner])

        self.map.positions[winner] += \
            lr * (point - self.map.positions[winner])

        for neighbor in self.map.neighbors(winner):

            self.winner_count[neighbor] += 1

            lr = 1.0 / (100 * self.winner_count[neighbor])

            self.map.positions[neighbor] += \
                lr * (point - self.map.positions[neighbor])


    def update_edge(self, s1, s2):

        # Increase age of all edges connected to s1
        for neighbor in self.map.neighbors(s1):

            self.map.graph[s1][neighbor] += 1
            self.map.graph[neighbor][s1] += 1

        # Reset existing edge or create a new one
        if self.map.has_edge(s1, s2):

            self.map.graph[s1][s2] = 0
            self.map.graph[s2][s1] = 0

        else:

            self.map.add_edge(s1, s2)

    def remove_old_edges(self, gmax):

        remove_edges = []

        visited = set()

        for a in self.map.graph:

            for b, age in self.map.graph[a].items():

                if (b, a) in visited:
                    continue

                visited.add((a, b))

                if age > gmax:
                    self.deleted_edge_ages.append(age)
                    remove_edges.append((a,b))

        for a, b in remove_edges:

            self.map.remove_edge(a, b)


    def process_frame(self, point_cloud):

        sampled = self.sample_points(point_cloud)

        for point in sampled:

            s1, s2, d1, d2 = self.winner_search(point)

            #
            # Case (a)
            #
            if d1 > self.vigilance:

                self.add_node(point)
                continue

            #
            # Case (b)
            #
            self.update_existing_node(
                point,
                s1
            )

            #
            # Case (c)
            #
            if d2 <= self.vigilance:

                self.update_edge(
                    s1,
                    s2
                )

        gmax = self.compute_gmax()
        self.remove_old_edges(gmax)

    def compute_gthr(self):

        ages = self.map.edge_ages()

        if len(ages) < 4:
            return np.inf

        q1 = np.percentile(ages, 25)
        q3 = np.percentile(ages, 75)

        iqr = q3 - q1

        return q3 + iqr


    def compute_gmax(self):

        current = self.map.edge_ages()

        gthr = self.compute_gthr()

        if len(self.deleted_edge_ages) == 0:
            return gthr

        gdel = np.mean(self.deleted_edge_ages)

        total = len(current) + len(self.deleted_edge_ages)

        weight = len(self.deleted_edge_ages) / total

        gmax = (
            gdel * weight +
            gthr * (1.0 - weight)
        )

        return gmax


class ScanDataset:

    def __init__(self, folder):

        self.folder = Path(folder)

        self.files = sorted(
            self.folder.glob("scan_*.csv")
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        return np.loadtxt(
            self.files[idx],
            delimiter=",",
            comments="#",
            skiprows=5
        ).astype(np.float32)


def visualize_map(atc: ATCDT):

    vis = o3d.visualization.Visualizer()
    vis.create_window("ATC-DT Result")

    #
    # Nodes
    #

    node_cloud = o3d.geometry.PointCloud()
    node_cloud.points = o3d.utility.Vector3dVector(
        atc.map.positions
    )
    node_cloud.paint_uniform_color([1, 0, 0])   # red

    vis.add_geometry(node_cloud)

    #
    # Edges
    #

    lines = o3d.geometry.LineSet()

    lines.points = o3d.utility.Vector3dVector(
        atc.map.positions
    )

    lines.lines = o3d.utility.Vector2iVector(
        np.asarray(atc.map.edges(), dtype=np.int32)
    )

    colors = np.tile(
        np.array([[0, 1, 0]]),
        (len(atc.map.edges()), 1)
    )

    lines.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(lines)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":

    dataset = ScanDataset(
        "/home/faza/Documents/pythonPrj/ta_test/dataset/scans"
    )

    atc = ATCDT(
        vigilance=0.3,
        lambda_points=6000
    )

    for i in range(len(dataset)):

        scan = dataset[i]

        print(
            f"Processing frame {i} "
            f"({len(scan)} points)"
        )

        atc.process_frame(scan)

    print()
    print("Finished.")
    print("Nodes :", atc.map.node_count)
    print("Edges :", atc.map.num_edges())

    visualize_map(atc)