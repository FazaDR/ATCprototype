from __future__ import annotations

from collections import defaultdict
from typing import Dict
import numpy as np
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
import time



class TopologicalMap:

    def __init__(self):
        self.node_count = 0
        self.next_node_id = 0
        self.capacity = 1024
        self.positions = np.empty(
            (self.capacity, 3),
            dtype=np.float32
        )

        self.normals = np.zeros(
            (self.capacity, 3),
            dtype=np.float32
        )

        self.slope_angles = np.zeros(
            self.capacity,
            dtype=np.float32
        )

        self.traversable = np.zeros(
            self.capacity,
            dtype=bool
        )

        self.contour = np.zeros(
            self.capacity,
            dtype=bool
        )

        self.node_ids = np.empty(
            self.capacity,
            dtype=np.int32
        )

        self.id_to_slot: Dict[int, int] = {}
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

        new_positions = np.empty(
            (new_capacity, 3),
            dtype=np.float32
        )

        new_node_ids = np.empty(
            new_capacity,
            dtype=np.int32
        )

        new_normals = np.zeros(
            (new_capacity, 3),
            dtype=np.float32
        )

        new_slope = np.zeros(
            new_capacity,
            dtype=np.float32
        )

        new_trav = np.zeros(
            new_capacity,
            dtype=bool
        )

        new_contour = np.zeros(
            new_capacity,
            dtype=bool
        )

        new_positions[:self.node_count] = (
            self.positions[:self.node_count]
        )

        new_node_ids[:self.node_count] = (
            self.node_ids[:self.node_count]
        )

        new_normals[:self.node_count] = (
            self.normals[:self.node_count]
        )

        new_slope[:self.node_count] = (
            self.slope_angles[:self.node_count]
        )

        new_trav[:self.node_count] = (
            self.traversable[:self.node_count]
        )

        new_contour[:self.node_count] = (
            self.contour[:self.node_count]
        )

        self.positions = new_positions
        self.node_ids = new_node_ids

        self.normals = new_normals
        self.slope_angles = new_slope
        self.traversable = new_trav
        self.contour = new_contour

        self.capacity = new_capacity

    def add_node(self, position: np.ndarray) -> int:
        if self.node_count >= self.capacity:
            self._grow_capacity()

        position = np.asarray(
            position,
            dtype=np.float32
        )

        slot = self.node_count

        node_id = self.next_node_id
        self.next_node_id += 1
        self.positions[slot] = position
        self.node_ids[slot] = node_id
        self.id_to_slot[node_id] = slot
        self.node_count += 1

        return node_id

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
        vigilance=1,
        lambda_points=5000,
        deg_max=np.deg2rad(30),
        theta_thr=np.pi,
        gravity=None,
        enable_attributes=False
    ):

        self.vigilance = vigilance

        self.lambda_points = lambda_points

        self.map = TopologicalMap()

        self.winner_count = np.zeros(
            self.map.capacity,
            dtype=np.int32
        )

        self.deleted_edge_count = 0
        self.deleted_edge_mean = 0.0

        self.enable_attributes = enable_attributes

        self.deg_max = deg_max

        self.theta_thr = theta_thr

        self.gravity = (
            np.array(
                [0.0, -1.0, 0.0],
                dtype=np.float32
            )
            if gravity is None
            else np.asarray(
                gravity,
                dtype=np.float32
            )
        )

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
        self.winner_count[slot] = 1

        return node_id

    def winner_search(
        self,
        point,
        candidates=None
    ):

        if self.map.node_count == 0:

            return None, None, np.inf, np.inf

        if candidates is None:

            slots = np.arange(
                self.map.node_count,
                dtype=np.int32
            )

        else:

            slots = np.asarray(
                [
                    self.map.get_slot(node_id)
                    for node_id in candidates
                ],
                dtype=np.int32
            )

            if len(slots) == 0:

                return None, None, np.inf, np.inf

        positions = self.map.positions[slots]

        dist = np.linalg.norm(
            positions - point,
            axis=1
        )

        if len(slots) == 1:

            slot = slots[0]

            return (
                int(self.map.node_ids[slot]),
                None,
                float(dist[0]),
                np.inf
            )

        order = np.argpartition(
            dist,
            1
        )

        s1_slot = slots[order[0]]
        s2_slot = slots[order[1]]

        d1 = dist[order[0]]
        d2 = dist[order[1]]

        if d2 < d1:

            s1_slot, s2_slot = (
                s2_slot,
                s1_slot
            )

            d1, d2 = (
                d2,
                d1
            )

        s1 = int(
            self.map.node_ids[s1_slot]
        )

        s2 = int(
            self.map.node_ids[s2_slot]
        )

        return s1, s2, float(d1), float(d2)

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


    def update_by_winners(
        self,
        point,
        s1,
        s2,
        d1,
        d2
    ):

        # --------------------------------------------------
        # Case (a)
        # Add new node
        # --------------------------------------------------

        if d1 > self.vigilance:

            new_node_id = self.add_node(
                point
            )

            return new_node_id

        # --------------------------------------------------
        # Case (b)
        # Update winner
        # --------------------------------------------------

        self.update_winner(
            point,
            s1
        )

        # --------------------------------------------------
        # Case (c)
        # Connect s1 and s2
        # --------------------------------------------------

        if (
            s2 is not None
            and
            d2 < self.vigilance
        ):

            self.map.add_edge(
                s1,
                s2
            )

            # Algorithm 2:
            # gs1,s2 <- 0
            self.map.graph[s1][s2] = 0
            self.map.graph[s2][s1] = 0

        # --------------------------------------------------
        # Update neighbors + age edges
        # --------------------------------------------------

        for neighbor in list(
            self.map.neighbors(s1)
        ):

            neighbor_slot = (
                self.map.get_slot(
                    neighbor
                )
            )

            lr = 1.0 / (
                100 *
                self.winner_count[
                    neighbor_slot
                ]
            )

            self.map.positions[
                neighbor_slot
            ] += (
                lr *
                (
                    point -
                    self.map.positions[
                        neighbor_slot
                    ]
                )
            )

            self.map.graph[s1][neighbor] += 1
            self.map.graph[neighbor][s1] += 1

        # --------------------------------------------------
        # Remove old edges
        # --------------------------------------------------

        gmax = self.compute_gmax(
            s1
        )

        self.remove_old_edges(
            s1,
            gmax
        )
        # Layer-1 attributes Upper layers have enable_attributes=False.

        if self.enable_attributes:

            self.estimate_normal(
                s1
            )

            self.estimate_traversability(
                s1
            )

            self.detect_contour(
                s1
            )

        return None

    def remove_old_edges(self, s1, gmax):

        remove_edges = []

        for neighbor in list(
            self.map.neighbors(s1)
        ):

            age = self.map.graph[s1][neighbor]

            if age > gmax:

                self.deleted_edge_count += 1

                self.deleted_edge_mean += (
                    age - self.deleted_edge_mean
                ) / self.deleted_edge_count

                remove_edges.append(
                    (s1, neighbor)
                )

        for a, b in remove_edges:

            self.map.remove_edge(a, b)


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

        if self.deleted_edge_count == 0:
            return gthr

        gdel = self.deleted_edge_mean

        total = (
            self.deleted_edge_count +
            len(gamma)
        )

        weight_deleted = (
            self.deleted_edge_count /
            total
        )

        gmax = (
            gdel * weight_deleted
            +
            gthr * (1.0 - weight_deleted)
        )

        return gmax

    def estimate_normal(self, s1):

        if not self.enable_attributes:
            return

        neighbor_ids = list(
            self.map.neighbors(s1)
        )

        if len(neighbor_ids) < 3:
            return

        s1_slot = self.map.get_slot(s1)

        p_s1 = self.map.positions[s1_slot]

        neighbor_slots = [
            self.map.get_slot(n)
            for n in neighbor_ids
        ]

        diffs = (
            self.map.positions[neighbor_slots]
            - p_s1
        )

        cov = diffs.T @ diffs

        eigvals, eigvecs = np.linalg.eigh(cov)

        # Eigenvector with smallest eigenvalue
        # is the estimated surface normal.
        self.map.normals[s1_slot] = eigvecs[:, 0]

    def estimate_traversability(self, s1):

        if not self.enable_attributes:
            return

        s1_slot = self.map.get_slot(s1)

        normal = self.map.normals[s1_slot]

        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            return

        cos_theta = np.clip(
            np.dot(
                normal,
                self.gravity
            ) / norm,
            -1.0,
            1.0
        )

        angle = np.arccos(cos_theta)
        angle = min(
            angle,
            np.pi - angle
        )

        self.map.slope_angles[s1_slot] = angle

        self.map.traversable[s1_slot] = (
            angle < self.deg_max
        )

    def detect_contour(
        self,
        s1,
        exclude_untraversable=True
    ):

        if not self.enable_attributes:
            return

        neighbor_ids = list(
            self.map.neighbors(s1)
        )

        s1_slot = self.map.get_slot(s1)

        if exclude_untraversable:

            confirmed = []

            for nid in neighbor_ids:

                nslot = self.map.get_slot(nid)

                has_normal = (
                    np.linalg.norm(
                        self.map.normals[nslot]
                    ) > 1e-8
                )

                if (
                    has_normal
                    and
                    self.map.traversable[nslot]
                ):
                    confirmed.append(nid)

            neighbor_ids = confirmed

        if len(neighbor_ids) < 2:

            self.map.contour[s1_slot] = True

            return

        p_s1 = self.map.positions[s1_slot]

        g = (
            self.gravity /
            np.linalg.norm(self.gravity)
        )

        ref = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        if abs(np.dot(ref, g)) > 0.9:

            ref = np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32
            )

        u = np.cross(
            g,
            ref
        )

        u /= np.linalg.norm(u)

        v = np.cross(
            g,
            u
        )

        angles = []

        for nid in neighbor_ids:

            diff = (
                self.map.positions[
                    self.map.get_slot(nid)
                ]
                - p_s1
            )

            diff_proj = (
                diff -
                np.dot(diff, g) * g
            )

            angles.append(
                np.arctan2(
                    np.dot(diff_proj, v),
                    np.dot(diff_proj, u)
                )
            )

        angles = np.sort(
            np.asarray(
                angles
            )
        )

        gaps = np.diff(
            angles
        )

        gaps = np.append(
            gaps,
            (angles[0] + 2 * np.pi)
            - angles[-1]
        )

        theta_max = np.max(
            gaps
        )

        self.map.contour[s1_slot] = (
            theta_max > self.theta_thr
        )


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



class Hierarchy:

    def __init__(self):

        # C_i^(ℓ)
        # children[1][3] = {4, 7, 9}
        #   Layer 2 node 3
        #   has Layer 1 children
        #   4, 7, 9

        self.children = defaultdict(
            lambda: defaultdict(set)
        )

        # Reverse relationship.
        # parent[1][7] = 3
        # means:
        #   Layer 1 node 7
        #   belongs to Layer 2 node 3.
        self.parent = defaultdict(dict)


    def add_relationship(
        self,
        upper_layer,
        parent_id,
        child_id
    ):

        self.children[
            upper_layer
        ][parent_id].add(
            child_id
        )

        self.parent[
            upper_layer
        ][child_id] = parent_id

    def get_children(
        self,
        upper_layer,
        parent_id
    ):

        return self.children[
            upper_layer
        ].get(
            parent_id,
            set()
        )

    def get_parent(
        self,
        upper_layer,
        child_id
    ):

        return self.parent[
            upper_layer
        ].get(
            child_id
        )


    def has_parent(
        self,
        upper_layer,
        child_id
    ):

        return child_id in self.parent[
            upper_layer
        ]



class MLATC:

    def __init__(
        self,
        base_vigilance=0.5,
        alpha=4.0,
        lambda_points=4000,
        seed=42
    ):
        
        self.base_vigilance = base_vigilance

        self.alpha = alpha

        self.lambda_points = lambda_points

        # Reproducible random generator
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.layers = []

        self.hierarchy = Hierarchy()

        self.add_layer()

    def add_layer(self):

        layer_index = len(self.layers)

        vigilance = (
            self.base_vigilance *
            self.alpha ** layer_index
        )

        enable_attributes = (
            layer_index == 0
        )

        layer = ATCDT(
            vigilance=vigilance,
            lambda_points=self.lambda_points,
            enable_attributes=enable_attributes
        )

        self.layers.append(layer)

        return layer_index

    def sample_points(self, points):

        n = len(points)

        if n <= self.lambda_points:

            idx = self.rng.permutation(n)

        else:

            idx = self.rng.choice(
                n,
                self.lambda_points,
                replace=False
            )

        return points[idx]

    def get_vigilance(self, layer_index):

        return (
            self.base_vigilance *
            self.alpha ** layer_index
        )

    def search_vigilance(self, layer_index):

        total = 0.0

        for i in range(layer_index + 1):

            total += self.layers[
                i
            ].vigilance

        return total

    def hierarchical_nns(self, point):

        num_layers = len(self.layers)

        # W^(ell)
        #
        # Python:
        # winner_sets[layer_index]
        winner_sets = [
            []
            for _ in range(num_layers)
        ]

        for layer_index in range(
            num_layers - 1,
            -1,
            -1
        ):

            layer = self.layers[
                layer_index
            ]

            if layer_index == num_layers - 1:

                candidates = (
                    layer.map.node_ids[
                        :layer.map.node_count
                    ].tolist()
                )

            else:

                upper_layer = layer_index + 1

                candidates = set()

                for parent_id in winner_sets[
                    upper_layer
                ]:

                    children = (
                        self.hierarchy.get_children(
                            upper_layer,
                            parent_id
                        )
                    )

                    candidates.update(
                        children
                    )

                rho_search = (
                    self.search_vigilance(
                        layer_index
                    )
                )

                filtered = []

                for node_id in candidates:

                    position = layer.map.get_position(
                        node_id
                    )

                    distance = np.linalg.norm(
                        point - position
                    )

                    if distance <= rho_search:

                        filtered.append(
                            (
                                node_id,
                                distance
                            )
                        )

                # Sort by distance
                filtered.sort(
                    key=lambda x: x[1]
                )

                winner_sets[
                    layer_index
                ] = [
                    node_id
                    for node_id, _ in filtered
                ]

                continue

            distances = []

            for node_id in candidates:

                position = layer.map.get_position(
                    node_id
                )

                distance = np.linalg.norm(
                    point - position
                )

                distances.append(
                    (
                        node_id,
                        distance
                    )
                )

            distances.sort(
                key=lambda x: x[1]
            )

            winner_sets[
                layer_index
            ] = [
                node_id
                for node_id, _ in distances
            ]

        return winner_sets

    def select_winners(
        self,
        layer_index,
        point,
        candidates
    ):

        layer = self.layers[
            layer_index
        ]

        return layer.winner_search(
            point,
            candidates
        )

    def assign_parent(
        self,
        lower_layer_index,
        child_id,
        parent_id
    ):

        upper_layer_index = (
            lower_layer_index + 1
        )

        self.hierarchy.add_relationship(
            upper_layer_index,
            parent_id,
            child_id
        )

    def expand_top_layer(self):

        old_top_index = (
            len(self.layers) - 1
        )

        old_top = self.layers[
            old_top_index
        ]

        if old_top.map.node_count != 2:

            raise RuntimeError(
                "Top layer expansion requires "
                "exactly two nodes."
            )

        first_id = int(
            old_top.map.node_ids[0]
        )

        second_id = int(
            old_top.map.node_ids[1]
        )

        first_position = (
            old_top.map.get_position(
                first_id
            ).copy()
        )

        second_position = (
            old_top.map.get_position(
                second_id
            ).copy()
        )

        new_layer_index = self.add_layer()

        new_layer = self.layers[
            new_layer_index
        ]

        root_id = new_layer.add_node(
            first_position
        )

        self.hierarchy.add_relationship(
            new_layer_index,
            root_id,
            first_id
        )

        s1, s2, d1, d2 = (
            new_layer.winner_search(
                second_position
            )
        )

        new_node_id = (
            new_layer.update_by_winners(
                second_position,
                s1,
                s2,
                d1,
                d2
            )
        )

        if new_node_id is None:

            parent_id = s1

        else:

            parent_id = new_node_id

        self.hierarchy.add_relationship(
            new_layer_index,
            parent_id,
            second_id
        )

        if new_layer.map.node_count == 2:
            return self.expand_top_layer()

        return new_layer_index

    def process_point(self, point):

        winner_sets = self.hierarchical_nns(point)

        layer_index = 0
        child_id = None

        while layer_index < len(self.layers):

            layer = self.layers[layer_index]

            candidates = winner_sets[layer_index]

            s1, s2, d1, d2 = layer.winner_search(
                point,
                candidates
            )

            new_node_id = layer.update_by_winners(
                point,
                s1,
                s2,
                d1,
                d2
            )

            winner_id = (
                new_node_id
                if new_node_id is not None
                else s1
            )

            if layer_index > 0 and child_id is not None:

                self.hierarchy.add_relationship(
                    layer_index,
                    winner_id,
                    child_id
                )

            if new_node_id is None:
                break

            if (
                layer_index == len(self.layers) - 1
                and
                layer.map.node_count == 2
            ):

                self.expand_top_layer()

                break

            child_id = new_node_id

            layer_index += 1

    def process_frame(self, point_cloud):

        sampled = self.sample_points(
            point_cloud
        )

        for point in sampled:

            self.process_point(
                point
            )

    

def visualize_mlatc(
    mlatc: MLATC,
    vertical_offset: float = 5.0
):

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        "MLATC Hierarchical Result"
    )


    for layer_index, layer in enumerate(
        mlatc.layers
    ):

        if layer.map.node_count == 0:
            continue


        offset = np.array(
            [0.0, layer_index * vertical_offset, 0.0],
            dtype=np.float32
        )

        positions = (
            layer.map.positions[
                :layer.map.node_count
            ].copy()
        )

        positions += offset

        node_cloud = o3d.geometry.PointCloud()

        node_cloud.points = (
            o3d.utility.Vector3dVector(
                positions
            )
        )

        if layer.enable_attributes:

            colors = mlatc_node_colors(
                layer
            )

            node_cloud.colors = (
                o3d.utility.Vector3dVector(
                    colors
                )
            )

        else:

            node_cloud.paint_uniform_color(
                [1, 0, 0]
            )

        vis.add_geometry(
            node_cloud
        )


        lines = o3d.geometry.LineSet()

        lines.points = (
            o3d.utility.Vector3dVector(
                positions
            )
        )

        edge_slots = []

        for a, b in layer.map.edges():

            if (
                a not in layer.map.id_to_slot
                or
                b not in layer.map.id_to_slot
            ):
                continue

            edge_slots.append([
                layer.map.get_slot(a),
                layer.map.get_slot(b)
            ])

        if len(edge_slots) > 0:

            lines.lines = (
                o3d.utility.Vector2iVector(
                    np.asarray(
                        edge_slots,
                        dtype=np.int32
                    )
                )
            )

            if layer.enable_attributes:
                edge_colors = [
                    (colors[sa] + colors[sb]) / 2.0
                    for sa, sb in edge_slots
                ]
            else:
                edge_colors = np.tile(
                    np.array([[0, 1, 0]]),
                    (len(edge_slots), 1)
                )

            lines.colors = (
                o3d.utility.Vector3dVector(
                    np.asarray(edge_colors, dtype=np.float32)
                )
            )

            vis.add_geometry(
                lines
            )

    vis.run()

    vis.destroy_window()

def mlatc_node_colors(
    layer
):

    n = layer.map.node_count

    colors = np.tile(
        np.array(
            [0.6, 0.6, 0.6],
            dtype=np.float32
        ),
        (n, 1)
    )

    normal_norms = np.linalg.norm(
        layer.map.normals[:n],
        axis=1
    )

    has_normal = (
        normal_norms > 1e-8
    )

    trav = layer.map.traversable[
        :n
    ]

    contour = layer.map.contour[
        :n
    ]

    # Traversable
    colors[
        has_normal & trav
    ] = [
        0.0,
        0.0,
        1.0
    ]

    # Traversable + contour
    colors[
        has_normal & trav & contour
    ] = [
        0.0,
        1.0,
        0.0
    ]

    # Untraversable
    colors[
        has_normal & ~trav
    ] = [
        1.0,
        0.0,
        0.0
    ]

    return colors

def save_mlatc_map(
    mlatc: MLATC,
    filename="mlatc_map.npz"
):
    data = {}

    for layer_index, layer in enumerate(mlatc.layers):

        n = layer.map.node_count

        data[f"layer_{layer_index}_positions"] = (
            layer.map.positions[:n].copy()
        )

        data[f"layer_{layer_index}_node_ids"] = (
            layer.map.node_ids[:n].copy()
        )

        data[f"layer_{layer_index}_normals"] = (
            layer.map.normals[:n].copy()
        )

        data[f"layer_{layer_index}_slope_angles"] = (
            layer.map.slope_angles[:n].copy()
        )

        data[f"layer_{layer_index}_traversable"] = (
            layer.map.traversable[:n].copy()
        )

        data[f"layer_{layer_index}_contour"] = (
            layer.map.contour[:n].copy()
        )

        edges = np.asarray(
            layer.map.edges(),
            dtype=np.int32
        )

        if edges.size == 0:
            edges = np.empty(
                (0, 2),
                dtype=np.int32
            )

        data[f"layer_{layer_index}_edges"] = edges

    hierarchy_rows = []

    for upper_layer, parent_children in mlatc.hierarchy.children.items():

        for parent_id, child_ids in parent_children.items():

            for child_id in child_ids:

                hierarchy_rows.append(
                    [upper_layer, parent_id, child_id]
                )

    data["hierarchy"] = np.asarray(
        hierarchy_rows,
        dtype=np.int32
    ).reshape(-1, 3)

    np.savez(
        filename,
        **data
    )

    print(
        f"MLATC map saved to: {filename}"
    )


def save_mlatc_map_and_plot(
    mlatc: MLATC,
    node_counts,
    layer1_node_counts,
    process_times,
    filename="mlatc_map.npz"
):

    save_mlatc_map(
        mlatc,
        filename
    )

    plt.figure()
    plt.plot(node_counts, process_times, marker="o")
    plt.xlabel("Total Nodes (All Layers)")
    plt.ylabel("Processing Time (s)")
    plt.title("Processing Time vs Total Nodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(layer1_node_counts, process_times, marker="o")
    plt.xlabel("Layer-1 Nodes")
    plt.ylabel("Processing Time (s)")
    plt.title("Processing Time vs Layer-1 Nodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    dataset = ScanDataset(
        "datasetLong/scans"
    )

    mlatc = MLATC(
        base_vigilance=0.5,
        alpha=4.0,
        lambda_points=3500,  # FYI long dataset have around like 5500k point (500)
        seed=42
    )

    process_times = []
    node_counts = []
    layer1_node_counts = []

    for i in range(len(dataset)):

        scan = dataset[i]

        print(
            f"Processing frame {i} "
            f"({len(scan)} points)"
        )

        start_time = time.perf_counter()

        mlatc.process_frame(
            scan
        )

        process_time = (
            time.perf_counter()
            - start_time
        )

        total_nodes = sum(
            layer.map.node_count
            for layer in mlatc.layers
        )

        layer1_nodes = (
            mlatc.layers[0].map.node_count
        )

        process_times.append(
            process_time
        )

        node_counts.append(
            total_nodes
        )

        layer1_node_counts.append(
            layer1_nodes
        )

        print(
            f"Frame {i} complete"
        )

        print(
            f"  Process time : "
            f"{process_time:.6f} s"
        )

        print(
            f"  Total nodes  : "
            f"{total_nodes}"
        )

        for layer_index, layer in enumerate(
            mlatc.layers,
            start=1
        ):

            print(
                f"  Layer {layer_index}: "
                f"Nodes = {layer.map.node_count}, "
                f"Edges = {layer.map.num_edges()}"
            )

    print()
    print("Total node count per frame:")
    print(node_counts)

    print()
    print("Layer-1 node count per frame:")
    print(layer1_node_counts)

    print()
    print("Process time per frame (seconds):")
    print(process_times)

    print()

    for layer_index, layer in enumerate(
        mlatc.layers,
        start=1
    ):

        print(
            f"Layer {layer_index}:"
        )

        print(
            "  Nodes :",
            layer.map.node_count
        )

        print(
            "  Edges :",
            layer.map.num_edges()
        )

    save_mlatc_map_and_plot(
        mlatc,
        node_counts,
        layer1_node_counts,
        process_times,
        "local_mlatc_show_0.5.npz"
    )

    visualize_mlatc(
        mlatc,
        5.0
    )