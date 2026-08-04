import os
import numpy as np
import pandas as pd
import open3d as o3d


def visualize_graph(graph):

    node_ids = list(graph.nodes.keys())

    if len(node_ids) == 0:
        return

    points = []
    colors = []

    id_to_idx = {}

    # --------------------------
    # NODES
    # --------------------------
    for idx, node_id in enumerate(node_ids):

        node = graph.nodes[node_id]
        id_to_idx[node_id] = idx

        points.append(node.pos)

        # color logic
        if node.contour == 1:
            colors.append([0, 0, 1])   # blue
        elif node.traversable == 1:
            colors.append([0, 1, 0])   # green
        else:
            colors.append([1, 0, 0])   # red


    points = np.array(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # --------------------------
    # EDGES
    # --------------------------
    lines = []

    for (i, j) in graph.edges.keys():

        if i in id_to_idx and j in id_to_idx:
            lines.append([id_to_idx[i], id_to_idx[j]])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    line_set.paint_uniform_color([0.5, 0.5, 0.5])

    # --------------------------
    # RENDER
    # --------------------------
    o3d.visualization.draw_geometries([pcd, line_set])


class ATCDTGraph:
    def __init__(self, v_thr):
        self.v_thr = v_thr
        self.nodes = {}      # id → Node
        self.edges = {}      # (i,j) → Edge
        self.next_id = 0
        self.g_max = 50
        self.edge_ages = []
        self.deleted_edges = []
        
    def update(self, p):

        if len(self.nodes) < 2:
            self._add_node(p)
            return

        s1, s2 = self._find_winners(p)

        node_s1 = self.nodes[s1]

        d1 = np.linalg.norm(p - node_s1.pos)

        # Step 3: vigilance check
        if d1 > self.v_thr:
            self._add_node(p)
            return

        # Step 4: update winner
        self._update_node(s1, p)

        # Step 5: update edges
        self._update_edges(s1, s2)

    def _find_winners(self, p):

        dists = []

        for node_id, node in self.nodes.items():
            d = np.linalg.norm(p - node.pos)
            dists.append((d, node_id))

        dists.sort()

        return dists[0][1], dists[1][1]

    def _add_node(self, p):

        node_id = self.next_id
        self.next_id += 1

        self.nodes[node_id] = Node(node_id, p)

        # connect to nearest existing node
        if len(self.nodes) > 1:

            nearest = None
            best = np.inf

            for i,n in self.nodes.items():

                if i == node_id:
                    continue

                d = np.linalg.norm(p-n.pos)

                if d < best:
                    best = d
                    nearest = i


            if nearest is not None:

                key=(min(node_id,nearest),
                    max(node_id,nearest))

                self.edges[key]=Edge()

    def _update_node(self, node_id, p):

        node = self.nodes[node_id]

        node.wins += 1

        lr = 1.0 / (10.0 * node.wins)

        node.pos = node.pos + lr * (p - node.pos)


    def _update_gmax(self):

        if len(self.edge_ages) < 5:
            self.g_max = 50
            return

        import numpy as np

        current = np.array(self.edge_ages[-100:])  # recent edges

        deleted = np.array(self.deleted_edges[-100:]) if len(self.deleted_edges) > 0 else current

        if len(deleted) == 0:
            self.g_max = 50
            return

        gamma_del = np.mean(deleted)

        q75 = np.percentile(current, 75)
        iqr = np.percentile(current, 75) - np.percentile(current, 25)

        ratio = len(deleted) / (len(deleted) + len(current) + 1e-6)

        self.g_max = (
            gamma_del * ratio +
            (q75 + iqr) * (1 - ratio)
        )

    def _update_edges(self, s1, s2):

        to_delete = []

        for (i, j), edge in self.edges.items():
            if i == s1 or j == s1:
                edge.age += 1

                if edge.age > self.g_max:
                    to_delete.append((i, j))

        for key in to_delete:
            self.edge_ages.append(self.edges[key].age)
            self.deleted_edges.append(self.edges[key].age)
            del self.edges[key]

        # second winner condition (important)
        if s2 is not None:

            key = (min(s1, s2), max(s1, s2))

            if key not in self.edges:
                self.edges[key] = Edge()

            self.edges[key].age = 0

            # ---- ATC-DT semantic layer ----
            node_i = self.nodes[s1]
            node_j = self.nodes[s2]

            if node_i.traversable is not None and node_j.traversable is not None:
                self.edges[key].tra = 1 if node_i.traversable == node_j.traversable else 0
            else:
                self.edges[key].tra = 0
        self._update_gmax()


        # update semantic information after topology exists

        self._compute_normal(s1)

        self._compute_traversability(
            s1,
            np.deg2rad(30)
        )

        self._compute_contour(s1)
        
        
    
    def _get_neighbors(self, node_id):
        neighbors = []

        for (i, j) in self.edges.keys():
            if i == node_id:
                neighbors.append(j)
            elif j == node_id:
                neighbors.append(i)

        return neighbors

    def _compute_normal(self, node_id):

        node = self.nodes[node_id]
        neighbors = self._get_neighbors(node_id)

        if len(neighbors) < 3:
            node.normal = np.array([0, 0, 1], dtype=np.float32)
            return node.normal

        pts = []

        for nb in neighbors:
            diff = self.nodes[nb].pos - node.pos
            pts.append(diff)

        pts = np.array(pts)

        if pts.shape[0] < 3:
            node.normal = np.array([0, 0, 1], dtype=np.float32)
            return node.normal

        F = pts.T @ pts

        eigvals, eigvecs = np.linalg.eigh(F)

        normal = eigvecs[:, np.argmin(eigvals)]

        # force normal upward
        if normal[2] < 0:
            normal = -normal

        node.normal = normal

        return normal

    def _compute_traversability(self, node_id, deg_max):

        node = self.nodes[node_id]

        if node.normal is None:
            node.traversable = 0
            return 0

        uz = np.array([0, 0, 1], dtype=np.float32)

        cos_theta = np.dot(node.normal, uz) / (
            np.linalg.norm(node.normal) * np.linalg.norm(uz)
        )

        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        deg = np.arccos(cos_theta)

        node.traversable = 1 if deg < deg_max else 0

        return node.traversable
    

    def _compute_contour(self, node_id, theta_thr=np.deg2rad(45)):

        node = self.nodes[node_id]
        neighbors = self._get_neighbors(node_id)

        if len(neighbors) < 3:
            return 0

        center = node.pos

        angles = []

        for nb in neighbors:
            vec = self.nodes[nb].pos - center
            vec[2] = 0  # project to horizontal plane

            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                continue

            vec = vec / norm

            angle = np.arctan2(vec[1], vec[0])
            angles.append(angle)

        if len(angles) < 3:
            return 0

        angles.sort()

        # circular gap check
        max_gap = 0
        for i in range(len(angles)):
            a1 = angles[i]
            a2 = angles[(i + 1) % len(angles)]

            gap = (a2 - a1) % (2 * np.pi)
            max_gap = max(max_gap, gap)

        node.contour = 1 if max_gap > theta_thr else 0
        return node.contour



class Node:
    def __init__(self, node_id, pos):
        self.id = node_id
        self.pos = pos
        self.normal = None
        self.traversable = None
        self.contour = 0
        self.wins = 1

class Edge:
    def __init__(self):
        self.age = 0
        self.tra = 0



class SyntheticLidarDataset:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.scans_path = os.path.join(
            dataset_path,
            "scans"
        )

        self.poses = pd.read_csv(
            os.path.join(
                dataset_path,
                "poses.csv"
            )
        )

    def __len__(self):

        return len(self.poses)

    def get_frame(self, frame_id):

        pose_row = self.poses.iloc[frame_id]

        scan_file = os.path.join(
            self.scans_path,
            f"scan_{frame_id:06d}.csv"
        )

        points = np.loadtxt(
            scan_file,
            delimiter=",",
            comments="#",
            skiprows=5
        )

        position = np.array([
            pose_row["x"],
            pose_row["y"],
            pose_row["z"]
        ], dtype=np.float32)

        timestamp = pose_row["timestamp"]

        return {
            "frame": frame_id,
            "timestamp": timestamp,
            "position": position,
            "points": points
        }

    def __iter__(self):

        for frame_id in range(len(self)):
            yield self.get_frame(frame_id)

def sample_points(points, num_samples=4000):

    n = len(points)

    if n <= num_samples:
        return points

    idx = np.random.choice(
        n,
        size=num_samples,
        replace=False
    )

    return points[idx]



graph = ATCDTGraph(v_thr=0.7)
dataset = SyntheticLidarDataset("dataset")

for frame in dataset:

    points = frame["points"]
    sensor_pos = frame["position"]

    sampled_points = sample_points(points, 4000)

    # points are already in world coordinates
    global_points = sampled_points

    # STEP 1: update topology
    for p in global_points:
        graph.update(p)

    # STEP 2: frame-level post processing (IMPORTANT)
    for node_id in graph.nodes:
        graph._compute_normal(node_id)
        graph._compute_traversability(node_id, np.deg2rad(30))
        graph._compute_contour(node_id)

    # STEP 3: optional logging / debugging
    contour_nodes = [n.id for n in graph.nodes.values() if n.contour == 1]
    traversable_nodes = [n.id for n in graph.nodes.values() if n.traversable == 1]

    print(
        "nodes:", len(graph.nodes),
        "edges:", len(graph.edges),
        "contour:", len(contour_nodes)
    )
visualize_graph(graph)

