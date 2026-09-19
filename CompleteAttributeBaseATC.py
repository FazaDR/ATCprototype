from __future__ import annotations

from collections import defaultdict
from typing import Dict
import numpy as np
from pathlib import Path
import open3d as o3d


class TopologicalMap:
    """
    Positional topological map used by ATC-DT.

    Corresponds to

        G = (V, h_pos, E_pos)

    where

        V         : node IDs
        positions : h_pos
        graph     : E_pos
    """

    def __init__(self):

        # -----------------------------
        # Node storage
        # -----------------------------

        # Number of currently active nodes
        self.node_count = 0

        # Next globally unique node ID
        self.next_node_id = 0

        # Allocated storage capacity
        self.capacity = 1024

        # Dense positional node storage.
        #
        # Only [0 : node_count] contains active nodes.
        self.positions = np.empty(
            (self.capacity, 3),
            dtype=np.float32
        )

        self.normals = np.zeros((self.capacity, 3), dtype=np.float32)       # h_nor
        self.slope_angles = np.zeros(self.capacity, dtype=np.float32)       # deg_i (radians)
        self.traversable = np.zeros(self.capacity, dtype=bool)              # h_tra
        self.contour = np.zeros(self.capacity, dtype=bool)                  # z_i

        # Stable node ID stored at each array slot.
        #
        # Example:
        #   node_ids[0] = 7
        #   node_ids[1] = 12
        #
        # These are NOT necessarily equal to the array index.
        self.node_ids = np.empty(
            self.capacity,
            dtype=np.int32
        )

        # Stable node ID -> array slot
        #
        # Example:
        #   id_to_slot[12] = 1
        self.id_to_slot: Dict[int, int] = {}

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

    def _grow_capacity(self):

        new_capacity = self.capacity * 2
        # //////////////////////////////
        new_positions = np.empty(
            (new_capacity, 3),
            dtype=np.float32
        )

        new_node_ids = np.empty(
                    new_capacity,
                    dtype=np.int32
                )        
        new_normals = np.zeros((new_capacity, 3), dtype=np.float32)
        new_slope = np.zeros(new_capacity, dtype=np.float32)
        new_trav = np.zeros(new_capacity, dtype=bool)
        new_contour = np.zeros(new_capacity, dtype=bool)

        # //////////////////////////////
        new_positions[:self.node_count] = self.positions[:self.node_count]

        new_normals[:self.node_count] = self.normals[:self.node_count]
        new_slope[:self.node_count] = self.slope_angles[:self.node_count]
        new_trav[:self.node_count] = self.traversable[:self.node_count]
        new_contour[:self.node_count] = self.contour[:self.node_count]

        new_node_ids[:self.node_count] = self.node_ids[:self.node_count]

        # //////////////////////////////
        self.positions = new_positions

        self.capacity = new_capacity
        self.normals = new_normals
        self.slope_angles = new_slope
        self.traversable = new_trav
        self.contour = new_contour
        self.node_ids = new_node_ids

    # ==========================================================
    # Node Operations
    # ==========================================================

    def add_node(self, position: np.ndarray) -> int:
        """
        Add a new positional node.

        Returns a stable node ID.
        """

        if self.node_count >= self.capacity:
            self._grow_capacity()

        position = np.asarray(
            position,
            dtype=np.float32
        )

        slot = self.node_count

        node_id = self.next_node_id
        self.next_node_id += 1

        # Store node position
        self.positions[slot] = position

        # Store stable node ID
        self.node_ids[slot] = node_id

        # ID -> slot
        self.id_to_slot[node_id] = slot

        self.node_count += 1

        return node_id

    # ==========================================================
    # Edge Operations
    # ==========================================================

    def has_edge(self, a: int, b: int) -> bool:

        return b in self.graph[a]

    def add_edge(self, a: int, b: int):

        self.graph[a][b] = 1
        self.graph[b][a] = 1

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

        return np.asarray(
            ages,
            dtype=np.float32
        )

    def edge_ages_of(self, node_id: int):
    
            return np.asarray(
                list(self.graph[node_id].values()),
                dtype=np.float32
            )

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

    def get_position(self, node_id: int) -> np.ndarray:

        slot = self.id_to_slot[node_id]

        return self.positions[slot]

    def get_slot(self, node_id: int) -> int:

        return self.id_to_slot[node_id]

    


class ATCDT:

    def __init__(
        self,
        vigilance=0.9,
        lambda_points=4000,
        deg_max=np.deg2rad(30),      # tune to your robot's max climbable slope
        theta_thr=np.pi,             # tune — start near 180°, a "half the compass is empty" gap
        gravity=None,
        exclude_untraversable_from_contour=True
    ):

        self.vigilance = vigilance
        self.lambda_points = lambda_points

        self.deleted_edge_ages = []
        self.map = TopologicalMap()
        self.winner_count = np.zeros(self.map.capacity, dtype=np.int32)

        self.deg_max = deg_max
        self.theta_thr = theta_thr
        # dataset is Y-up (see lidar_generation.py) — gravity points down, i.e. -Y
        self.gravity = (
            np.array([0.0, -1.0, 0.0], dtype=np.float32)
            if gravity is None else np.asarray(gravity, dtype=np.float32)
        )
        self.exclude_untraversable_from_contour = exclude_untraversable_from_contour


    def initialize(self, point_cloud):

        idx = np.random.choice(
            len(point_cloud),
            2,
            replace=False
        )

        self.add_node(point_cloud[idx[0]])
        self.add_node(point_cloud[idx[1]])

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

        old_capacity = self.map.capacity

        node_id = self.map.add_node(point)

        if self.map.capacity != old_capacity:

            new_winner_count = np.zeros(
                self.map.capacity,
                dtype=np.int32
            )

            new_winner_count[
                :self.map.node_count - 1
            ] = self.winner_count[
                :self.map.node_count - 1
            ]

            self.winner_count = new_winner_count

        slot = self.map.get_slot(node_id)

        # Paper Eq. (3): M_N+1 = 1
        self.winner_count[slot] = 1

        return node_id

    def winner_search(self, point):

        if self.map.node_count == 0:
            return None, None, np.inf, np.inf

        active_positions = self.map.positions[:self.map.node_count]

        dist = np.linalg.norm(
            active_positions - point,
            axis=1
        )

        if self.map.node_count == 1:

            slot = 0

            node_id = self.map.node_ids[slot]

            return (
                int(node_id),
                None,
                dist[slot],
                np.inf
            )

        order = np.argpartition(
            dist,
            1
        )

        s1_slot = order[0]
        s2_slot = order[1]

        if dist[s2_slot] < dist[s1_slot]:

            s1_slot, s2_slot = s2_slot, s1_slot

        d1 = dist[s1_slot]
        d2 = dist[s2_slot]

        s1 = int(
            self.map.node_ids[s1_slot]
        )

        s2 = int(
            self.map.node_ids[s2_slot]
        )

        return s1, s2, d1, d2

    def update_neighbors(self, point, winner):

        for neighbor in self.map.neighbors(winner):

            neighbor_slot = self.map.get_slot(neighbor)

            # Eq. (5)
            lr = 1.0 / (
                100 * self.winner_count[neighbor_slot]
            )

            self.map.positions[neighbor_slot] += (
                lr *
                (point - self.map.positions[neighbor_slot])
            )

    def update_winner(self, point, winner):

        winner_slot = self.map.get_slot(winner)

        # M_s1 = M_s1 + 1
        self.winner_count[winner_slot] += 1

        # Eq. (4)
        lr = 1.0 / (
            10 * self.winner_count[winner_slot]
        )

        self.map.positions[winner_slot] += (
            lr *
            (point - self.map.positions[winner_slot])
        )

    def remove_old_edges(self, s1, gmax):

        remove_edges = []

        for neighbor in list(
            self.map.neighbors(s1)
        ):

            age = self.map.graph[s1][neighbor]

            if age > gmax:

                self.deleted_edge_ages.append(age)

                remove_edges.append(
                    (s1, neighbor)
                )

        for a, b in remove_edges:

            self.map.remove_edge(a, b)

    def process_frame(self, point_cloud):

        sampled = self.sample_points(point_cloud)

        if self.map.node_count == 0:

            if len(sampled) < 2:
                return

            self.initialize(sampled)

            # Start learning after initialization
            # sampled = sampled[2:] <-- use if node init are ordered

        for point in sampled:

            s1, s2, d1, d2 = self.winner_search(point)

            # --------------------------------------------------
            # Case (a): Add new node
            # ds1 > vigilance
            # --------------------------------------------------

            if d1 >= self.vigilance:

                self.add_node(point)

                # Paper Step 3:
                # after adding a node, go to the next input
                continue

            # --------------------------------------------------
            # Case 2: Update s1
            # --------------------------------------------------

            self.update_winner(
                point,
                s1
            )

            # --------------------------------------------------
            # Case 3:
            # Update neighbors only when s2
            # is within vigilance
            # --------------------------------------------------

            if d2 < self.vigilance:

                self.update_neighbors(
                    point,
                    s1
                )




            # --------------------------------------------------
            # Step 5: Age edges connected to s1
            # --------------------------------------------------

            for neighbor in list(
                self.map.neighbors(s1)
            ):

                self.map.graph[s1][neighbor] += 1
                self.map.graph[neighbor][s1] += 1

            # --------------------------------------------------
            # Step 4/5: normal, traversability, contour for s1
            # --------------------------------------------------
            self.estimate_normal(s1)
            self.estimate_traversability(s1)
            self.detect_contour(s1)


            # --------------------------------------------------
            # Case (c): Add/reset s1-s2 edge
            # ds2 < vigilance
            # --------------------------------------------------

            if d2 < self.vigilance:

                if self.map.has_edge(s1, s2):

                    self.map.graph[s1][s2] = 1
                    self.map.graph[s2][s1] = 1

                else:

                    self.map.add_edge(
                        s1,
                        s2
                    )

            # --------------------------------------------------
            # Step 6:
            # Calculate gmax for THIS s1 and remove old edges
            # --------------------------------------------------

            gmax = self.compute_gmax(s1)

            self.remove_old_edges(
                s1,
                gmax
            )

    def compute_gthr(self, gamma):

        if len(gamma) == 0:
            return np.inf

        q3 = np.percentile(
            gamma,
            75
        )

        q1 = np.percentile(
            gamma,
            25
        )

        iqr = q3 - q1

        return q3 + iqr

    def compute_gmax(self, s1):

        gamma = self.map.edge_ages_of(s1)

        if len(gamma) == 0:
            return np.inf

        gthr = self.compute_gthr(gamma)

        if len(self.deleted_edge_ages) == 0:
            return gthr

        gdel = np.mean(
            self.deleted_edge_ages
        )

        total = (
            len(self.deleted_edge_ages) +
            len(gamma)
        )

        weight_deleted = (
            len(self.deleted_edge_ages) /
            total
        )

        gmax = (
            gdel * weight_deleted
            +
            gthr * (1.0 - weight_deleted)
        )

        return gmax

    def estimate_normal(self, s1):
        neighbor_ids = list(self.map.neighbors(s1))

        if len(neighbor_ids) < 3:
            return  # not enough neighbors for a stable PCA plane fit

        s1_slot = self.map.get_slot(s1)
        p_s1 = self.map.positions[s1_slot]

        neighbor_slots = [self.map.get_slot(n) for n in neighbor_ids]
        diffs = self.map.positions[neighbor_slots] - p_s1  # Eq. (11)

        cov = diffs.T @ diffs

        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order

        self.map.normals[s1_slot] = eigvecs[:, 0]  # eigenvector of min eigenvalue


    def estimate_traversability(self, s1):
        s1_slot = self.map.get_slot(s1)
        normal = self.map.normals[s1_slot]

        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return  # normal not yet available

        cos_theta = np.clip(np.dot(normal, self.gravity) / norm, -1.0, 1.0)
        deg = np.arccos(cos_theta)  # Eq. (12), radians

        # PCA normals have arbitrary sign — fold to the acute angle from vertical
        deg = min(deg, np.pi - deg)

        self.map.slope_angles[s1_slot] = deg
        self.map.traversable[s1_slot] = deg < self.deg_max  # Eq. (13)

    def detect_contour(self, s1, exclude_untraversable=True):
        neighbor_ids = list(self.map.neighbors(s1))
        s1_slot = self.map.get_slot(s1)

        if exclude_untraversable:
            confirmed = []
            for nid in neighbor_ids:
                nslot = self.map.get_slot(nid)
                has_normal = np.linalg.norm(self.map.normals[nslot]) > 1e-8
                # only neighbors CONFIRMED traversable count as "covering" a direction.
                # untraversable neighbors, and neighbors not yet classified
                # (no normal computed), are treated as unknown/open — same as
                # having no neighbor there at all.
                if has_normal and self.map.traversable[nslot]:
                    confirmed.append(nid)
            neighbor_ids = confirmed

        if len(neighbor_ids) < 2:
            self.map.contour[s1_slot] = True  # surrounded by nothing traversable -> boundary
            return

        p_s1 = self.map.positions[s1_slot]
        g = self.gravity / np.linalg.norm(self.gravity)

        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(ref, g)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = np.cross(g, ref); u /= np.linalg.norm(u)
        v = np.cross(g, u)

        angles = []
        for nid in neighbor_ids:
            diff = self.map.positions[self.map.get_slot(nid)] - p_s1
            diff_proj = diff - np.dot(diff, g) * g
            angles.append(np.arctan2(np.dot(diff_proj, v), np.dot(diff_proj, u)))

        angles = np.sort(np.asarray(angles))
        gaps = np.diff(angles)
        gaps = np.append(gaps, (angles[0] + 2 * np.pi) - angles[-1])

        theta_max = np.max(gaps)
        self.map.contour[s1_slot] = theta_max > self.theta_thr




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

    vis.create_window(
        "ATC-DT Result"
    )

    # -----------------------------
    # Nodes
    # -----------------------------

    node_cloud = o3d.geometry.PointCloud()

    node_cloud.points = o3d.utility.Vector3dVector(atc.map.positions[:atc.map.node_count])

    node_cloud.paint_uniform_color(
        [1, 0, 0]
    )

    vis.add_geometry(node_cloud)

    # -----------------------------
    # Edges
    # -----------------------------

    lines = o3d.geometry.LineSet()

    lines.points = o3d.utility.Vector3dVector(atc.map.positions[:atc.map.node_count])

    edge_slots = []

    for a, b in atc.map.edges():

        if (
            a not in atc.map.id_to_slot
            or
            b not in atc.map.id_to_slot
        ):
            continue

        edge_slots.append([
            atc.map.get_slot(a),
            atc.map.get_slot(b)
        ])

    lines.lines = o3d.utility.Vector2iVector(np.asarray(edge_slots,dtype=np.int32))

    colors = np.tile(
        np.array([[0, 1, 0]]),
        (len(edge_slots), 1)
    )

    lines.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(lines)

    vis.run()

    vis.destroy_window()


def node_colors(atc: ATCDT):
    n = atc.map.node_count
    colors = np.tile(np.array([0.6, 0.6, 0.6], dtype=np.float32), (n, 1))  # gray: no normal yet

    normal_norms = np.linalg.norm(atc.map.normals[:n], axis=1)
    has_normal = normal_norms > 1e-8
    trav = atc.map.traversable[:n]
    contour = atc.map.contour[:n]

    colors[has_normal & trav] = [0.0, 0.0, 1.0]              # blue: traversable, not contour
    colors[has_normal & trav & contour] = [0.0, 1.0, 0.0]    # green: traversable AND contour
    colors[has_normal & ~trav] = [1.0, 0.0, 0.0]             # red: untraversable (contour or not)

    return colors

def normal_lines(atc: ATCDT, length=0.3):
    """
    Build points/segments for visualizing each node's normal axis
    as a line centered on the node, extending length/2 in both
    directions along h_nor. Sign-agnostic: +n and -n draw identically.
    """
    n = atc.map.node_count

    normal_norms = np.linalg.norm(atc.map.normals[:n], axis=1)
    has_normal = normal_norms > 1e-8

    idx = np.where(has_normal)[0]

    if len(idx) == 0:
        return None, None

    base_points = atc.map.positions[idx]
    unit_normals = atc.map.normals[idx] / normal_norms[idx, None]

    half = length / 2.0
    end_a = base_points - unit_normals * half
    end_b = base_points + unit_normals * half

    points = np.empty((2 * len(idx), 3), dtype=np.float32)
    points[0::2] = end_a
    points[1::2] = end_b

    lines = np.array(
        [[2 * k, 2 * k + 1] for k in range(len(idx))],
        dtype=np.int32
    )

    return points, lines

def visualize_map_attributes(atc: ATCDT, show_normals=True, normal_length=0.3):

    vis = o3d.visualization.Visualizer()
    vis.create_window("ATC-DT Attributes")

    n = atc.map.node_count
    colors = node_colors(atc)

    # -----------------------------
    # Nodes
    # -----------------------------

    node_cloud = o3d.geometry.PointCloud()
    node_cloud.points = o3d.utility.Vector3dVector(atc.map.positions[:n])
    node_cloud.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(node_cloud)

    # -----------------------------
    # Edges — color follows endpoint average
    # -----------------------------

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(atc.map.positions[:n])

    edge_slots = []
    edge_colors = []

    for a, b in atc.map.edges():

        if a not in atc.map.id_to_slot or b not in atc.map.id_to_slot:
            continue

        sa = atc.map.get_slot(a)
        sb = atc.map.get_slot(b)

        edge_slots.append([sa, sb])
        edge_colors.append((colors[sa] + colors[sb]) / 2.0)

    lines.lines = o3d.utility.Vector2iVector(
        np.asarray(edge_slots, dtype=np.int32)
    )
    lines.colors = o3d.utility.Vector3dVector(
        np.asarray(edge_colors, dtype=np.float32)
    )
    vis.add_geometry(lines)

    # -----------------------------
    # Normal direction lines
    # -----------------------------

    if show_normals:

        points, segs = normal_lines(atc, length=normal_length)

        if points is not None:

            normal_set = o3d.geometry.LineSet()
            normal_set.points = o3d.utility.Vector3dVector(points)
            normal_set.lines = o3d.utility.Vector2iVector(segs)
            normal_set.paint_uniform_color([0.0, 0.0, 0.0])  # yellow

            vis.add_geometry(normal_set)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":

    dataset = ScanDataset(
        "datasetStatic/scans",
    )

    atc = ATCDT(
        vigilance=0.5,
        lambda_points=9999999
    )

    for i in range(len(dataset)):

        scan = dataset[i]

        print(
            f"Processing frame {i} "
            f"({len(scan)} points)"
        )

        atc.process_frame(scan)

        print(
            f"Frame {i} complete | "
            f"Nodes: {atc.map.node_count} | "
            f"Edges: {atc.map.num_edges()}"
        )

    print()
    print("Finished.")
    print("Nodes :", atc.map.node_count)
    print("Edges :", atc.map.num_edges())

    visualize_map_attributes(atc)  # new attribute view

