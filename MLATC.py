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

        new_positions = np.empty(
            (new_capacity, 3),
            dtype=np.float32
        )

        new_node_ids = np.empty(
            new_capacity,
            dtype=np.int32
        )

        new_positions[:self.node_count] = self.positions[:self.node_count]

        new_node_ids[:self.node_count] = self.node_ids[:self.node_count]

        self.positions = new_positions
        self.node_ids = new_node_ids

        self.capacity = new_capacity

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
        vigilance=0.9
    ):

        self.vigilance = vigilance

        self.map = TopologicalMap()

        self.winner_count = np.zeros(
            self.map.capacity,
            dtype=np.int32
        )

        self.deleted_edge_count = 0
        self.deleted_edge_mean = 0.0

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

    def winner_search(
        self,
        point,
        candidates=None
    ):

        if self.map.node_count == 0:

            return None, None, np.inf, np.inf

        # ---------------------------------
        # Candidate nodes
        # ---------------------------------

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

        # ---------------------------------
        # Distance
        # ---------------------------------

        positions = self.map.positions[slots]

        dist = np.linalg.norm(
            positions - point,
            axis=1
        )

        # ---------------------------------
        # One candidate
        # ---------------------------------

        if len(slots) == 1:

            slot = slots[0]

            return (
                int(self.map.node_ids[slot]),
                None,
                float(dist[0]),
                np.inf
            )

        # ---------------------------------
        # Two nearest candidates
        # ---------------------------------

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
        """
        Corresponds to Algorithm 2:
        UPDATEBYWINNERS

        Returns
        -------
        new_node_id : int or None
            ID of newly created node, if a new node was added.
        """

        # --------------------------------------------------
        # Step 1-2:
        # Add new node if ds1 > vigilance
        # --------------------------------------------------

        if d1 > self.vigilance:

            new_node_id = self.add_node(point)

            return new_node_id

        # --------------------------------------------------
        # Step 4-5:
        # Update first winner
        # --------------------------------------------------

        self.update_winner(
            point,
            s1
        )

        # --------------------------------------------------
        # Step 6-8:
        # Connect s1 and s2 if ds2 < vigilance
        # --------------------------------------------------

        if s2 is not None and d2 < self.vigilance:

            self.map.add_edge(
                s1,
                s2
            )

            # Algorithm 2:
            # gs1,s2 <- 0
            #
            # Your add_edge() initializes age to 1,
            # so explicitly reset it here.
            self.map.graph[s1][s2] = 0
            self.map.graph[s2][s1] = 0

        # --------------------------------------------------
        # Step 10-12:
        # Update neighbors and age edges
        # --------------------------------------------------

        for neighbor in list(
            self.map.neighbors(s1)
        ):

            # Attenuated neighbor update
            neighbor_slot = self.map.get_slot(neighbor)

            lr = 1.0 / (
                100 * self.winner_count[neighbor_slot]
            )

            self.map.positions[neighbor_slot] += (
                lr *
                (point - self.map.positions[neighbor_slot])
            )

            # Age edge
            self.map.graph[s1][neighbor] += 1
            self.map.graph[neighbor][s1] += 1

        # --------------------------------------------------
        # Step 14-15:
        # Remove old edges and update gmax
        # --------------------------------------------------

        gmax = self.compute_gmax(s1)

        self.remove_old_edges(
            s1,
            gmax
        )

        return None

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


# class NPZPositionDataset:

#     def __init__(self, path):

#         self.path = Path(path)

#         if not self.path.exists():
#             raise FileNotFoundError(
#                 f"File not found: {self.path}"
#             )

#     def __len__(self):

#         return 1

#     def __getitem__(self, idx):

#         if idx != 0:
#             raise IndexError(
#                 "NPZPositionDataset contains only one file"
#             )

#         data = np.load(
#             self.path
#         )

#         if "positions" not in data:

#             raise KeyError(
#                 f"'positions' not found in {self.path}"
#             )

#         return data["positions"].astype(
#             np.float32
#         )


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

    # def remove_relationship(
    #     self,
    #     upper_layer,
    #     child_id
    # ):

    #     parent_id = self.parent[
    #         upper_layer
    #     ].pop(
    #         child_id,
    #         None
    #     )

    #     if parent_id is None:
    #         return

    #     self.children[
    #         upper_layer
    #     ][parent_id].discard(
    #         child_id
    #     )



class MLATC:

    def __init__(
        self,
        base_vigilance=0.5,
        alpha=4.0,
        lambda_points=4000
    ):
        
        self.base_vigilance = base_vigilance

        self.alpha = alpha

        self.lambda_points = lambda_points

        self.layers = []

        self.hierarchy = Hierarchy()

        self.add_layer()

    def add_layer(self):

        layer_index = len(self.layers) + 1
        vigilance = self.base_vigilance * self.alpha ** (layer_index - 1)
        layer = ATCDT(vigilance=vigilance)
        self.layers.append(layer)
        return len(self.layers) - 1

    def sample_points(self, points):

        n = len(points)

        if n <= self.lambda_points:

            idx = np.random.permutation(n)

        else:

            idx = np.random.choice(
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

        # --------------------------------------------------
        # Search from top layer -> bottom layer
        # --------------------------------------------------

        for layer_index in range(
            num_layers - 1,
            -1,
            -1
        ):

            layer = self.layers[
                layer_index
            ]

            # ----------------------------------------------
            # Top layer
            # ----------------------------------------------

            if layer_index == num_layers - 1:

                candidates = (
                    layer.map.node_ids[
                        :layer.map.node_count
                    ].tolist()
                )

            # ----------------------------------------------
            # Lower layers
            # ----------------------------------------------

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

                # ------------------------------------------
                # Eq. (18)
                # ------------------------------------------

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

            # ----------------------------------------------
            # Top layer sorting
            # ----------------------------------------------

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

        # --------------------------------------------------
        # Existing top-layer nodes
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Create new upper layer
        # --------------------------------------------------

        new_layer_index = self.add_layer()

        new_layer = self.layers[
            new_layer_index
        ]

        # --------------------------------------------------
        # Inherit first node as root
        # --------------------------------------------------

        root_id = new_layer.add_node(
            first_position
        )

        # --------------------------------------------------
        # First old top node becomes child
        # --------------------------------------------------

        self.hierarchy.add_relationship(
            new_layer_index,
            root_id,
            first_id
        )

        # --------------------------------------------------
        # Second old top node becomes input
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Determine parent of second node
        # --------------------------------------------------

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
        child_id = None  # newly-created node from the layer below, pending a parent link

        while layer_index < len(self.layers):
            layer = self.layers[layer_index]
            candidates = winner_sets[layer_index]

            s1, s2, d1, d2 = layer.winner_search(point, candidates)
            new_node_id = layer.update_by_winners(point, s1, s2, d1, d2)

            winner_id = new_node_id if new_node_id is not None else s1

            if layer_index > 0 and child_id is not None:
                self.hierarchy.add_relationship(layer_index, winner_id, child_id)

            if new_node_id is None:
                break

            if layer_index == len(self.layers) - 1 and layer.map.node_count == 2:
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


if __name__ == "__main__":

    dataset = ScanDataset(
        "datasetStatic/scans"
    )

    mlatc = MLATC(
        base_vigilance=0.1,
        alpha=4.0,
        lambda_points=999999
    )

    for i in range(len(dataset)):

        scan = dataset[i]

        print(
            f"Processing frame {i} "
            f"({len(scan)} points)"
        )

        mlatc.process_frame(
            scan
        )

        print(
            f"Frame {i} complete"
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
    print("Finished.")

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

    visualize_mlatc(mlatc,5.0)