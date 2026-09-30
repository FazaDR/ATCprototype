from __future__ import annotations

from collections import defaultdict
from typing import Dict
from pathlib import Path

import numpy as np
import open3d as o3d


# ==========================================================
# Topological Map
# ==========================================================

class TopologicalMap:
    """
    Positional topological map.

        G = (V, {h_i}, E)

    where:

        V          : node IDs
        positions  : node positions
        graph      : positional edges
    """

    def __init__(self):

        # Number of currently active nodes
        self.node_count = 0

        # Next globally unique node ID
        self.next_node_id = 0

        # Allocated storage capacity
        self.capacity = 1024

        # Dense position storage.
        # Only [0 : node_count] contains active nodes.
        self.positions = np.empty(
            (self.capacity, 3),
            dtype=np.float32
        )

        # Stable node ID stored at each array slot.
        #
        # Example:
        #   node_ids[0] = 7
        #   node_ids[1] = 12
        #
        # Node IDs are not necessarily equal to array indices.
        self.node_ids = np.empty(
            self.capacity,
            dtype=np.int32
        )

        # Stable node ID -> array slot
        #
        # Example:
        #   id_to_slot[12] = 1
        self.id_to_slot: Dict[int, int] = {}

        # Positional edge graph.
        #
        # graph[a][b] = edge age
        self.graph: Dict[int, Dict[int, int]] = defaultdict(dict)

    # ==========================================================
    # Capacity
    # ==========================================================

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

        new_positions[:self.node_count] = (
            self.positions[:self.node_count]
        )

        new_node_ids[:self.node_count] = (
            self.node_ids[:self.node_count]
        )

        self.positions = new_positions
        self.node_ids = new_node_ids

        self.capacity = new_capacity

    # ==========================================================
    # Node Operations
    # ==========================================================

    def add_node(self, position: np.ndarray) -> int:
        """
        Add a new positional node.

        Returns:
            Stable node ID.
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

        # Store position
        self.positions[slot] = position

        # Store stable ID
        self.node_ids[slot] = node_id

        # ID -> slot mapping
        self.id_to_slot[node_id] = slot

        self.node_count += 1

        return node_id

    def remove_node(self, node_id: int):
        """
        Remove a node using swap-delete.

        The node ID is permanently retired.

        The final active array slot is moved into the
        deleted node's slot.
        """

        if node_id not in self.id_to_slot:
            return

        slot = self.id_to_slot[node_id]

        # Remove graph connections
        for neighbor in list(self.graph[node_id]):
            self.remove_edge(node_id, neighbor)

        self.graph.pop(node_id, None)

        # Last active node
        last_slot = self.node_count - 1

        # Move the last node into the removed node's slot
        if slot != last_slot:

            moved_id = self.node_ids[last_slot]

            self.positions[slot] = (
                self.positions[last_slot]
            )

            self.node_ids[slot] = moved_id

            # Update moved node's slot
            self.id_to_slot[moved_id] = slot

        # Remove deleted node
        del self.id_to_slot[node_id]

        # Decrease active node count
        self.node_count -= 1

        return slot, last_slot

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

    def get_position(self, node_id: int) -> np.ndarray:

        slot = self.id_to_slot[node_id]

        return self.positions[slot]

    def get_slot(self, node_id: int) -> int:

        return self.id_to_slot[node_id]

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


# ==========================================================
# ATC-DT
# ==========================================================

class ATCDT:

    def __init__(
        self,
        vigilance=0.5,
        lambda_points=4000,
        free_area_sectors=24,
        horizontal_band=0.15,
        free_area_distance=0.5
    ):

        self.vigilance = vigilance
        self.lambda_points = lambda_points

        self.free_area_sectors = free_area_sectors
        self.horizontal_band = horizontal_band
        self.free_area_distance = free_area_distance

        self.deleted_edge_count = 0
        self.deleted_edge_mean = 0.0

        # Positional topological map
        self.map = TopologicalMap()

        # Winner count is stored per array slot
        self.winner_count = np.zeros(
            self.map.capacity,
            dtype=np.int32
        )

    def initialize(self, point_cloud):

        idx = np.random.choice(
            len(point_cloud),
            2,
            replace=False
        )

        self.add_node(point_cloud[idx[0]])
        self.add_node(point_cloud[idx[1]])

    # ==========================================================
    # Point Sampling
    # ==========================================================


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

    # ==========================================================
    # Node Operations
    # ==========================================================

    def add_node(self, point):

        old_capacity = self.map.capacity

        node_id = self.map.add_node(point)

        # Resize winner count if map capacity increased
        if self.map.capacity != old_capacity:

            new_winner_count = np.zeros(
                self.map.capacity,
                dtype=np.int32
            )

            new_winner_count[:self.map.node_count - 1] = (
                self.winner_count[
                    :self.map.node_count - 1
                ]
            )

            self.winner_count = new_winner_count

        slot = self.map.get_slot(node_id)

        # Paper Eq. (3):
        # M_N+1 = 1
        self.winner_count[slot] = 1

        return node_id

    def remove_node(self, node_id):

        if node_id not in self.map.id_to_slot:
            return

        slot = self.map.get_slot(node_id)

        last_slot = self.map.node_count - 1

        # Move winner count together with swap-delete
        if slot != last_slot:

            self.winner_count[slot] = (
                self.winner_count[last_slot]
            )

        self.map.remove_node(node_id)

    # ==========================================================
    # Winner Search
    # ==========================================================

    def winner_search(self, point):

        if self.map.node_count == 0:

            return (
                None,
                None,
                np.inf,
                np.inf
            )

        active_positions = (
            self.map.positions[
                :self.map.node_count
            ]
        )

        dist = np.linalg.norm(
            active_positions - point,
            axis=1
        )

        # Only one node exists
        if self.map.node_count == 1:

            slot = 0

            node_id = self.map.node_ids[slot]

            return (
                int(node_id),
                None,
                dist[slot],
                np.inf
            )

        # Find two nearest nodes
        order = np.argpartition(
            dist,
            1
        )

        s1_slot = order[0]
        s2_slot = order[1]

        # Ensure s1 is nearest
        if dist[s2_slot] < dist[s1_slot]:

            s1_slot, s2_slot = (
                s2_slot,
                s1_slot
            )

        d1 = dist[s1_slot]
        d2 = dist[s2_slot]

        s1 = int(
            self.map.node_ids[s1_slot]
        )

        s2 = int(
            self.map.node_ids[s2_slot]
        )

        return (
            s1,
            s2,
            d1,
            d2
        )

    # ==========================================================
    # Node Adaptation
    # ==========================================================

    def update_existing_node(
        self,
        point,
        winner
    ):

        winner_slot = self.map.get_slot(winner)

        # Increment winner count
        self.winner_count[winner_slot] += 1

        # Winner learning rate
        lr = 1.0 / (
            10 *
            self.winner_count[winner_slot]
        )

        self.map.positions[winner_slot] += (
            lr * (
                point -
                self.map.positions[winner_slot]
            )
        )

        # Update neighbors
        for neighbor in self.map.neighbors(winner):

            neighbor_slot = (
                self.map.get_slot(neighbor)
            )

            lr = 1.0 / (
                100 *
                self.winner_count[neighbor_slot]
            )

            self.map.positions[neighbor_slot] += (
                lr * (
                    point -
                    self.map.positions[neighbor_slot]
                )
            )

    # ==========================================================
    # Edge Aging
    # ==========================================================

    def remove_old_edges(self, s1, gmax):
    
            remove_edges = []
    
            for neighbor in list(
                self.map.neighbors(s1)
            ):
    
                age = self.map.graph[s1][neighbor]
    
                if age > gmax:
    
                    # ------------------------------------------
                    # Update running mean of deleted edge ages
                    # ------------------------------------------
    
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

        # No deleted-edge history yet
        if self.deleted_edge_count == 0:
            return gthr

        # Mean age of all previously deleted edges
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

    # ==========================================================
    # Free Area Detection
    # ==========================================================

    def horizontal_scan(
        self,
        point_cloud,
        robot_position
    ):

        point_cloud = np.asarray(
            point_cloud,
            dtype=np.float32
        )

        robot_position = np.asarray(
            robot_position,
            dtype=np.float32
        )

        vertical_difference = np.abs(
            point_cloud[:, 1] -
            robot_position[1]
        )

        mask = (
            vertical_difference <=
            self.horizontal_band
        )

        return point_cloud[mask]

    def find_proximity_points(
        self,
        point_cloud,
        robot_position
    ):

        scan = self.horizontal_scan(
            point_cloud,
            robot_position
        )

        if len(scan) == 0:
            return []

        dx = (
            scan[:, 0] -
            robot_position[0]
        )

        dz = (
            scan[:, 2] -
            robot_position[2]
        )

        distances = np.sqrt(
            dx * dx +
            dz * dz
        )

        angles = np.arctan2(
            dz,
            dx
        )

        angles = np.mod(
            angles,
            2.0 * np.pi
        )

        sector_width = (
            2.0 * np.pi /
            self.free_area_sectors
        )

        sector_indices = (
            angles / sector_width
        ).astype(np.int32)

        proximity_points = []

        for sector in range(
            self.free_area_sectors
        ):

            mask = (
                sector_indices == sector
            )

            if not np.any(mask):
                continue

            sector_indices_local = (
                np.where(mask)[0]
            )

            closest_local = (
                sector_indices_local[
                    np.argmin(
                        distances[
                            sector_indices_local
                        ]
                    )
                ]
            )

            proximity_points.append(
                scan[closest_local]
            )

        return proximity_points

    def node_in_free_area(
        self,
        node_position,
        proximity_points,
        robot_position
    ):

        if len(proximity_points) == 0:
            return False

        node = np.asarray(
            node_position,
            dtype=np.float32
        )

        robot = np.asarray(
            robot_position,
            dtype=np.float32
        )

        node_2d = np.array([
            node[0],
            node[2]
        ])

        robot_2d = np.array([
            robot[0],
            robot[2]
        ])

        for r in proximity_points:

            r_2d = np.array([
                r[0],
                r[2]
            ])

            q = r_2d - robot_2d

            norm_q = np.linalg.norm(q)

            if norm_q < 1e-6:
                continue

            q /= norm_q

            value = np.dot(
                q,
                r_2d - node_2d
            )

            if value <= 0:
                return False

        return True

    # ==========================================================
    # Free Area Node Deletion
    # ==========================================================

    def delete_free_area_nodes(
        self,
        point_cloud,
        robot_position
    ):

        proximity_points = (
            self.find_proximity_points(
                point_cloud,
                robot_position
            )
        )

        if len(proximity_points) < 2:
            return

        nodes_to_delete = []

        active_positions = (
            self.map.positions[
                :self.map.node_count
            ]
        )

        active_ids = (
            self.map.node_ids[
                :self.map.node_count
            ]
        )

        for slot, node_id in enumerate(active_ids):

            node_position = (
                active_positions[slot]
            )

            # Only nodes inside the detected
            # free area are deletion candidates.
            if not self.node_in_free_area(
                node_position,
                proximity_points,
                robot_position
            ):
                continue

            # Eq. (8):
            # Find closest point cloud point
            # to this node.
            distances = np.linalg.norm(
                point_cloud - node_position,
                axis=1
            )

            min_distance = np.min(
                distances
            )

            if (
                min_distance >
                self.free_area_distance
            ):

                nodes_to_delete.append(
                    int(node_id)
                )

        # Delete after iteration so that
        # array slots remain valid during testing.
        for node_id in nodes_to_delete:

            self.remove_node(node_id)

    # ==========================================================
    # Main Processing
    # ==========================================================

    def process_frame(
        self,
        point_cloud,
        robot_position,
        visualizer=None,
        update_every=50
    ):

        sampled = self.sample_points(
            point_cloud
        )

        for i, point in enumerate(sampled):

            s1, s2, d1, d2 = (
                self.winner_search(point)
            )

            # ----------------------------------------------
            # Case (a):
            # Add a new node when ds1 > vigilance
            # ----------------------------------------------

            if d1 > self.vigilance:

                self.add_node(point)

                continue

            # ----------------------------------------------
            # Cases (b) and (c):
            # Update existing winner
            # ----------------------------------------------

            self.update_existing_node(
                point,
                s1
            )

            # ----------------------------------------------
            # Age edges connected to s1
            # ----------------------------------------------

            for neighbor in list(
                self.map.neighbors(s1)
            ):

                self.map.graph[s1][neighbor] += 1

                self.map.graph[neighbor][s1] += 1

            # ----------------------------------------------
            # Case (c):
            # Add or reset edge between s1 and s2
            # ----------------------------------------------

            if d2 <= self.vigilance:

                if self.map.has_edge(s1, s2):

                    self.map.graph[s1][s2] = 0
                    self.map.graph[s2][s1] = 0

                else:

                    self.map.add_edge(s1, s2)

            # ----------------------------------------------
            # Visualization update
            # ----------------------------------------------

            if (
                visualizer is not None and
                i % update_every == 0
            ):

                visualizer.update(
                    self,
                    sampled
                )

            # ----------------------------------------------
            # Update gmax and remove old edges
            # ----------------------------------------------

            gmax = self.compute_gmax(s1)

            self.remove_old_edges(s1, gmax)

        # ----------------------------------------------
        # Free-area node deletion
        # ----------------------------------------------

        self.delete_free_area_nodes(
            point_cloud,
            robot_position
        )

        if visualizer is not None:

            visualizer.update(
                self,
                sampled
            )


# ==========================================================
# Live Visualizer
# ==========================================================

class ATCVisualizer:

    def __init__(self, first_scan):

        self.vis = o3d.visualization.Visualizer()

        self.vis.create_window(
            "ATC"
        )

        # Scan cloud
        self.scan_cloud = (
            o3d.geometry.PointCloud()
        )

        self.scan_cloud.points = (
            o3d.utility.Vector3dVector(
                first_scan
            )
        )

        self.scan_cloud.paint_uniform_color(
            [0.8, 0.8, 0.8]
        )

        # Node cloud
        self.node_cloud = (
            o3d.geometry.PointCloud()
        )

        # Edge line set
        self.line_set = (
            o3d.geometry.LineSet()
        )

        self.vis.add_geometry(
            self.scan_cloud
        )

        self.vis.add_geometry(
            self.node_cloud
        )

        self.vis.add_geometry(
            self.line_set
        )

    def update(self, atc, scan=None):

        # ---------------------------------
        # Scan
        # ---------------------------------

        if scan is not None:

            self.scan_cloud.points = (
                o3d.utility.Vector3dVector(
                    scan
                )
            )

            self.scan_cloud.paint_uniform_color(
                [0.75, 0.75, 0.75]
            )

            self.vis.update_geometry(
                self.scan_cloud
            )

        # ---------------------------------
        # Nodes
        # ---------------------------------

        if atc.map.node_count == 0:

            pts = np.empty(
                (0, 3),
                dtype=np.float32
            )

        else:

            pts = atc.map.positions[
                :atc.map.node_count
            ]

        self.node_cloud.points = (
            o3d.utility.Vector3dVector(
                pts
            )
        )

        self.node_cloud.paint_uniform_color(
            [1, 0, 0]
        )

        # ---------------------------------
        # Edges
        # ---------------------------------

        self.line_set.points = (
            o3d.utility.Vector3dVector(
                pts
            )
        )

        edge_slots = []

        for a, b in atc.map.edges():

            if (
                a not in atc.map.id_to_slot or
                b not in atc.map.id_to_slot
            ):
                continue

            edge_slots.append([
                atc.map.get_slot(a),
                atc.map.get_slot(b)
            ])

        if len(edge_slots) == 0:

            edge_slots = np.empty(
                (0, 2),
                dtype=np.int32
            )

        else:

            edge_slots = np.asarray(
                edge_slots,
                dtype=np.int32
            )

        self.line_set.lines = (
            o3d.utility.Vector2iVector(
                edge_slots
            )
        )

        colors = np.zeros(
            (len(edge_slots), 3),
            dtype=np.float64
        )

        colors[:] = [0, 1, 0]

        self.line_set.colors = (
            o3d.utility.Vector3dVector(
                colors
            )
        )

        self.vis.update_geometry(
            self.node_cloud
        )

        self.vis.update_geometry(
            self.line_set
        )

        self.vis.poll_events()

        self.vis.update_renderer()

    def close(self):

        self.vis.destroy_window()


# ==========================================================
# Dataset
# ==========================================================

class ScanDataset:

    def __init__(
        self,
        folder,
        pose_file
    ):

        self.folder = Path(folder)

        self.files = sorted(
            self.folder.glob(
                "scan_*.csv"
            )
        )

        self.poses = np.loadtxt(
            pose_file,
            delimiter=",",
            skiprows=1
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

    def get_pose(self, idx):

        return self.poses[
            idx,
            2:5
        ].astype(np.float32)


# ==========================================================
# Final Map Visualization
# ==========================================================

def visualize_map(atc: ATCDT):

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        "ATC-DT Result"
    )

    # ----------------------------------------------
    # Nodes
    # ----------------------------------------------

    node_cloud = (
        o3d.geometry.PointCloud()
    )

    node_cloud.points = (
        o3d.utility.Vector3dVector(
            atc.map.positions[
                :atc.map.node_count
            ]
        )
    )

    node_cloud.paint_uniform_color(
        [1, 0, 0]
    )

    vis.add_geometry(
        node_cloud
    )

    # ----------------------------------------------
    # Edges
    # ----------------------------------------------

    lines = (
        o3d.geometry.LineSet()
    )

    lines.points = (
        o3d.utility.Vector3dVector(
            atc.map.positions[
                :atc.map.node_count
            ]
        )
    )

    edge_slots = []

    for a, b in atc.map.edges():

        if (
            a not in atc.map.id_to_slot or
            b not in atc.map.id_to_slot
        ):
            continue

        edge_slots.append([
            atc.map.get_slot(a),
            atc.map.get_slot(b)
        ])

    edge_slots = np.asarray(
        edge_slots,
        dtype=np.int32
    )

    if len(edge_slots) == 0:

        edge_slots = np.empty(
            (0, 2),
            dtype=np.int32
        )

    lines.lines = (
        o3d.utility.Vector2iVector(
            edge_slots
        )
    )

    colors = np.tile(
        np.array([[0, 1, 0]]),
        (len(edge_slots), 1)
    )

    lines.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    vis.add_geometry(
        lines
    )

    vis.run()

    vis.destroy_window()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    dataset = ScanDataset(
        "datasetFleeting/scans",
        "datasetFleeting/poses.csv"
    )

    scan = dataset[0]

    atc = ATCDT(
        vigilance=0.5,
        lambda_points=6000
    )

    visualizer = ATCVisualizer(
        scan
    )

    visualizer.update(
        atc,
        scan
    )

    for i in range(len(dataset)):

        scan = dataset[i]

        pose = dataset.get_pose(i)

        print(
            f"Processing frame {i} "
            f"({len(scan)} points) "
            f"pose={pose}"
        )

        if atc.map.node_count == 0:

            atc.initialize(scan)

        atc.process_frame(
            scan,
            pose,
            visualizer,
            update_every=20
        )

    print()
    print("Finished.")
    print("Nodes :", atc.map.node_count)
    print("Edges :", atc.map.num_edges())

    visualizer.close()

    visualize_map(atc)