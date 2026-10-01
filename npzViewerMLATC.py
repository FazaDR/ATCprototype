import argparse

import numpy as np
import open3d as o3d


DEFAULT_NODE_COLOR = np.array([1.0, 0.0, 0.0], dtype=np.float32)
EDGE_COLOR = np.array([0.0, 1.0, 0.0], dtype=np.float32)
PARENT_CHILD_COLOR = np.array([0.75, 0.75, 0.75], dtype=np.float32)

# GLFW key codes (arrow keys aren't plain ASCII)
GLFW_KEY_UP = 265
GLFW_KEY_DOWN = 264


def load_layers(npz_path):

    data = np.load(npz_path)

    layers = []
    layer_index = 0

    while f"layer_{layer_index}_positions" in data:

        positions = data[f"layer_{layer_index}_positions"].astype(np.float32)
        node_ids = data[f"layer_{layer_index}_node_ids"]
        normals = data[f"layer_{layer_index}_normals"].astype(np.float32)
        traversable = data[f"layer_{layer_index}_traversable"].astype(bool)
        contour = data[f"layer_{layer_index}_contour"].astype(bool)
        edges = data[f"layer_{layer_index}_edges"]

        # Saved edges are (node_id, node_id) pairs -> convert to
        # slot indices so they can index straight into `positions`.
        id_to_slot = {
            int(node_id): slot
            for slot, node_id in enumerate(node_ids)
        }

        edge_slots = [
            [id_to_slot[int(a)], id_to_slot[int(b)]]
            for a, b in edges
            if int(a) in id_to_slot and int(b) in id_to_slot
        ]

        layers.append({
            "positions": positions,
            "node_ids": node_ids,
            "normals": normals,
            "traversable": traversable,
            "contour": contour,
            "edge_slots": np.asarray(edge_slots, dtype=np.int32)
                if edge_slots else np.empty((0, 2), dtype=np.int32),
        })

        layer_index += 1

    if not layers:
        raise ValueError(f"No layer_*_positions arrays found in {npz_path}")

    hierarchy = data["hierarchy"] if "hierarchy" in data else np.empty((0, 3), dtype=np.int32)

    return layers, hierarchy.astype(np.int32)


def node_colors(layer):
    """Same color scheme as `mlatc_node_colors` in the main script."""

    n = len(layer["positions"])

    colors = np.tile(np.array([0.6, 0.6, 0.6], dtype=np.float32), (n, 1))

    has_normal = np.linalg.norm(layer["normals"], axis=1) > 1e-8
    trav = layer["traversable"]
    contour = layer["contour"]

    colors[has_normal & trav] = [0.0, 0.0, 1.0]            # traversable
    colors[has_normal & trav & contour] = [0.0, 1.0, 0.0]  # + contour
    colors[has_normal & ~trav] = [1.0, 0.0, 0.0]            # untraversable

    return colors


class OffsetViewer:

    def __init__(self, layers, hierarchy, offset, step):

        self.layers = layers
        self.hierarchy = hierarchy
        self.offset = offset
        self.initial_offset = offset
        self.step = step
        self.parent_child_visible = False

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("MLATC NPZ Viewer (adjustable offset)")

        self.point_clouds = []
        self.line_sets = []
        self.parent_child_set = o3d.geometry.LineSet()

        for layer in layers:

            has_attrs = np.any(
                np.linalg.norm(layer["normals"], axis=1) > 1e-8
            )

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(layer["positions"])

            if has_attrs:
                pcd.colors = o3d.utility.Vector3dVector(node_colors(layer))
            else:
                pcd.paint_uniform_color(DEFAULT_NODE_COLOR.tolist())

            self.vis.add_geometry(pcd)
            self.point_clouds.append(pcd)

            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(layer["positions"])

            if len(layer["edge_slots"]) > 0:
                lines.lines = o3d.utility.Vector2iVector(layer["edge_slots"])
                lines.colors = o3d.utility.Vector3dVector(
                    np.tile(EDGE_COLOR, (len(layer["edge_slots"]), 1))
                )

            self.vis.add_geometry(lines)
            self.line_sets.append(lines)

        self._update_parent_child_geometry()

        for key in (ord("="), GLFW_KEY_UP):
            self.vis.register_key_callback(key, self._increase_offset)

        for key in (ord("-"), GLFW_KEY_DOWN):
            self.vis.register_key_callback(key, self._decrease_offset)

        self.vis.register_key_callback(ord("0"), self._reset_offset)
        self.vis.register_key_callback(ord("h"), self._toggle_parent_child)
        self.vis.register_key_callback(ord("H"), self._toggle_parent_child)

        self._apply_offset()

    def _increase_offset(self, vis):
        self.offset += self.step
        self._apply_offset()
        return False

    def _decrease_offset(self, vis):
        self.offset = max(0.0, self.offset - self.step)
        self._apply_offset()
        return False

    def _reset_offset(self, vis):
        self.offset = self.initial_offset
        self._apply_offset()
        return False

    def _toggle_parent_child(self, vis):
        self.parent_child_visible = not self.parent_child_visible

        if self.parent_child_visible:
            self.vis.add_geometry(self.parent_child_set)
        else:
            self.vis.remove_geometry(self.parent_child_set)

        if self.parent_child_visible:
            self._apply_offset()

        print(
            "Parent-child relationships: "
            f"{'shown' if self.parent_child_visible else 'hidden'}"
        )
        return False

    def _update_parent_child_geometry(self):
        id_to_slot = [
            {
                int(node_id): slot
                for slot, node_id in enumerate(layer["node_ids"])
            }
            for layer in self.layers
        ]

        points = []
        lines = []
        relationship_layers = []

        for upper_layer, parent_id, child_id in self.hierarchy:

            upper_layer = int(upper_layer)
            parent_id = int(parent_id)
            child_id = int(child_id)
            lower_layer = upper_layer - 1

            if (
                upper_layer < 1
                or upper_layer >= len(self.layers)
                or parent_id not in id_to_slot[upper_layer]
                or child_id not in id_to_slot[lower_layer]
            ):
                continue

            parent_slot = id_to_slot[upper_layer][parent_id]
            child_slot = id_to_slot[lower_layer][child_id]
            points.extend([
                self.layers[upper_layer]["positions"][parent_slot],
                self.layers[lower_layer]["positions"][child_slot]
            ])
            line_index = len(points) - 2
            lines.append([line_index, line_index + 1])
            relationship_layers.append((upper_layer, lower_layer))

        self.parent_child_base_points = np.asarray(
            points,
            dtype=np.float32
        ).reshape(-1, 3)
        self.parent_child_layers = relationship_layers

        if lines:
            self.parent_child_set.points = o3d.utility.Vector3dVector(
                self.parent_child_base_points
            )
            self.parent_child_set.lines = o3d.utility.Vector2iVector(
                np.asarray(lines, dtype=np.int32)
            )
            self.parent_child_set.colors = o3d.utility.Vector3dVector(
                np.tile(
                    PARENT_CHILD_COLOR,
                    (len(lines), 1)
                )
            )

    def _apply_offset(self):

        print(f"Vertical offset: {self.offset:.2f}")

        for layer_index, layer in enumerate(self.layers):

            shift = np.array(
                [0.0, layer_index * self.offset, 0.0], dtype=np.float32
            )

            shifted = layer["positions"] + shift

            self.point_clouds[layer_index].points = (
                o3d.utility.Vector3dVector(shifted)
            )

            self.line_sets[layer_index].points = (
                o3d.utility.Vector3dVector(shifted)
            )

            self.vis.update_geometry(self.point_clouds[layer_index])
            self.vis.update_geometry(self.line_sets[layer_index])

        if self.parent_child_visible:
            shifted = self.parent_child_base_points.copy()

            for line_index, (upper_layer, lower_layer) in enumerate(
                self.parent_child_layers
            ):
                point_start = 2 * line_index
                shifted[point_start, 1] += upper_layer * self.offset
                shifted[point_start + 1, 1] += lower_layer * self.offset

            self.parent_child_set.points = o3d.utility.Vector3dVector(shifted)
            self.vis.update_geometry(self.parent_child_set)

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


def main():

    parser = argparse.ArgumentParser(
        description="View an MLATC .npz map with a live-adjustable "
                    "vertical offset between layers."
    )

    parser.add_argument(
        "npz_path", nargs="?", default="mapsShow/local_mlatc_show_0.5.npz",
        help="Path to the .npz map (default: maps/local_mlatc_long_1.npz)"
    )
    parser.add_argument(
        "--offset", type=float, default=10.0,
        help="Starting vertical offset between layers (default: 5.0)"
    )
    parser.add_argument(
        "--step", type=float, default=0.5,
        help="Offset change per key press (default: 0.5)"
    )

    args = parser.parse_args()

    layers, hierarchy = load_layers(args.npz_path)

    print(f"Loaded {len(layers)} layer(s) from {args.npz_path}")

    for i, layer in enumerate(layers):
        print(
            f"  Layer {i}: {len(layer['positions'])} nodes, "
            f"{len(layer['edge_slots'])} edges"
        )

    print()
    print("Controls (click the window first):")
    print("  =  / Up     increase offset")
    print("  -  / Down   decrease offset")
    print("  0           reset offset")
    print("  H           toggle parent-child relationships")
    print("  Q / Esc     quit")
    print()

    OffsetViewer(
        layers,
        hierarchy,
        offset=args.offset,
        step=args.step
    ).run()


if __name__ == "__main__":
    main()