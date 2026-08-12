import open3d as o3d
import numpy as np
import os
import time


# ==========================================================
# Configuration
# ==========================================================

DATASET_DIR = "dataset/scans"

ROOM_SIZE = 8.0
ROOM_HEIGHT = 3.0
WALL_THICKNESS = 0.2


# ==========================================================
# Environment
# ==========================================================

def create_environment(with_divider=True):

    floor = o3d.geometry.TriangleMesh.create_box(
        width=ROOM_SIZE,
        height=0.2,
        depth=ROOM_SIZE
    )

    wall1 = o3d.geometry.TriangleMesh.create_box(
        width=ROOM_SIZE,
        height=ROOM_HEIGHT,
        depth=WALL_THICKNESS
    )

    wall2 = o3d.geometry.TriangleMesh.create_box(
        width=WALL_THICKNESS,
        height=ROOM_HEIGHT,
        depth=ROOM_SIZE
    )

    wall3 = o3d.geometry.TriangleMesh.create_box(
        width=ROOM_SIZE,
        height=ROOM_HEIGHT,
        depth=WALL_THICKNESS
    )

    wall4 = o3d.geometry.TriangleMesh.create_box(
        width=WALL_THICKNESS,
        height=ROOM_HEIGHT,
        depth=ROOM_SIZE
    )

    wall1.translate([0, 0, 0])
    wall2.translate([ROOM_SIZE, 0, 0])
    wall3.translate([0, 0, ROOM_SIZE])
    wall4.translate([0, 0, 0])

    meshes = [
        floor,
        wall1,
        wall2,
        wall3,
        wall4
    ]

    if with_divider:

        divider = o3d.geometry.TriangleMesh.create_box(
            width=0.15,
            height=ROOM_HEIGHT,
            depth=5.0
        )

        divider.translate([
            ROOM_SIZE / 2 - 0.075,
            0,
            1.5
        ])

        meshes.append(divider)

    environment = meshes[0]

    for mesh in meshes[1:]:
        environment += mesh

    environment.compute_vertex_normals()

    return environment


# ==========================================================
# Load scan
# ==========================================================

def load_scan(frame_id):

    filename = os.path.join(
        DATASET_DIR,
        f"scan_{frame_id:06d}.csv"
    )

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    # Your CSV contains:
    #
    # # timestamp=...
    # # sensor_x=...
    # # sensor_y=...
    # # sensor_z=...
    # x,y,z
    #
    # Therefore skip the first 4 metadata lines
    # and the fifth line is the column header.

    points = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=5
    )

    if points.ndim == 1:
        points = points.reshape(1, 3)

    return points


# ==========================================================
# Load pose
# ==========================================================

def load_poses():

    filename = "dataset/poses.csv"

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True
    )

    return data


# ==========================================================
# Create LiDAR visualization
# ==========================================================

def create_lidar(position):

    lidar = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.15
    )

    lidar.compute_vertex_normals()

    lidar.paint_uniform_color(
        [1.0, 0.0, 1.0]
    )

    lidar.translate(position)

    return lidar


# ==========================================================
# Create coordinate frame
# ==========================================================

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0,
    origin=[0, 0, 0]
)


# ==========================================================
# Load poses
# ==========================================================

poses = load_poses()

num_frames = len(poses)

print("Number of frames:", num_frames)


# ==========================================================
# Initial state
# ==========================================================

frame_id = 0

points = load_scan(frame_id)

sensor_position = np.array([
    poses["x"][frame_id],
    poses["y"][frame_id],
    poses["z"][frame_id]
])


# ==========================================================
# Point cloud
# ==========================================================

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    points
)

pcd.paint_uniform_color(
    [1.0, 1.0, 0.0]
)


# ==========================================================
# Environment
# ==========================================================

environment = create_environment(
    with_divider=True
)


# ==========================================================
# LiDAR
# ==========================================================

lidar = create_lidar(
    sensor_position
)


# ==========================================================
# Visualization
# ==========================================================

vis = o3d.visualization.Visualizer()

vis.create_window(
    window_name="Dataset Visualization",
    width=1280,
    height=720
)

vis.add_geometry(environment)
vis.add_geometry(pcd)
vis.add_geometry(lidar)
vis.add_geometry(axis)


# ==========================================================
# Camera
# ==========================================================

ctr = vis.get_view_control()

ctr.set_front([
    0.0,
    1.0,
    0.0
])

ctr.set_up([
    0.0,
    0.0,
    1.0
])

ctr.set_lookat([
    ROOM_SIZE / 2,
    ROOM_HEIGHT / 2,
    ROOM_SIZE / 2
])

ctr.set_zoom(0.35)


# ==========================================================
# Animation
# ==========================================================

for frame_id in range(num_frames):

    print(
        f"\nFrame {frame_id}"
    )

    # ------------------------------------------------------
    # Load scan
    # ------------------------------------------------------

    points = load_scan(frame_id)

    print(
        "Points:",
        len(points)
    )

    # ------------------------------------------------------
    # Load sensor position
    # ------------------------------------------------------

    sensor_position = np.array([
        poses["x"][frame_id],
        poses["y"][frame_id],
        poses["z"][frame_id]
    ])

    print(
        "LiDAR position:",
        sensor_position
    )

    # ------------------------------------------------------
    # Change environment at frame 6
    # ------------------------------------------------------

    if frame_id == 6:

        print(
            ">>> DIVIDER REMOVED"
        )

        vis.remove_geometry(
            environment,
            reset_bounding_box=False
        )

        environment = create_environment(
            with_divider=False
        )

        vis.add_geometry(
            environment,
            reset_bounding_box=False
        )

    # ------------------------------------------------------
    # Update point cloud
    # ------------------------------------------------------

    pcd.points = o3d.utility.Vector3dVector(
        points
    )

    vis.update_geometry(
        pcd
    )

    # ------------------------------------------------------
    # Move LiDAR
    # ------------------------------------------------------

    lidar.translate(
        sensor_position -
        np.asarray(lidar.get_center())
    )

    vis.update_geometry(
        lidar
    )

    # ------------------------------------------------------
    # Render
    # ------------------------------------------------------

    vis.poll_events()
    vis.update_renderer()

    time.sleep(1.0)


# ==========================================================
# Keep window open
# ==========================================================

print("\nFinished.")

vis.run()

vis.destroy_window()
