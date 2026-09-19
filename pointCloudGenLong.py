import open3d as o3d
import numpy as np
import time
import os


# ==========================================================
# LiDAR raycast
# ==========================================================

def lidar_scan(scene, origin, dirs):

    origins = np.repeat(
        origin[None, :],
        len(dirs),
        axis=0
    )

    rays = np.hstack([origins, dirs])

    rays = o3d.core.Tensor(
        rays,
        dtype=o3d.core.Dtype.Float32
    )

    ans = scene.cast_rays(rays)

    t_hit = ans["t_hit"].numpy()

    valid = np.isfinite(t_hit)

    hit_points = (
        origins[valid]
        + dirs[valid] * t_hit[valid, None]
    )

    distances = np.linalg.norm(
        hit_points - origin,
        axis=1
    )

    MIN_RANGE = 0.1
    MAX_RANGE = 20.0

    mask = (
        (distances >= MIN_RANGE) &
        (distances <= MAX_RANGE)
    )

    hit_points = hit_points[mask]

    return hit_points


# ==========================================================
# LONG DATASET PARAMETERS
# ==========================================================

# Corridor width
ROOM_WIDTH = 6.0

# Very long environment
ROOM_LENGTH = 100.0

# Wall dimensions
ROOM_HEIGHT = 3.0
WALL_THICKNESS = 0.2

# ----------------------------------------------------------
# LiDAR movement
# ----------------------------------------------------------

# Forward movement direction = +Z
START_Z = 1.5

# How far LiDAR moves every frame
FORWARD_INCREMENT = 10

# LiDAR lateral position:
#
#  0.0  = exactly in the middle
#  +     = shifted toward +X wall
#  -     = shifted toward -X wall
#
LATERAL_OFFSET = 0.0

LIDAR_HEIGHT = 2.0

# How long to scan
END_Z = ROOM_LENGTH - 1.5


# ==========================================================
# ENVIRONMENT
# ==========================================================

# ----------------------------------------------------------
# Floor
# ----------------------------------------------------------

floor = o3d.geometry.TriangleMesh.create_box(
    width=ROOM_WIDTH,
    height=0.2,
    depth=ROOM_LENGTH
)

floor.translate([
    0,
    0,
    0
])


# ----------------------------------------------------------
# Left side wall
# ----------------------------------------------------------

wall_left = o3d.geometry.TriangleMesh.create_box(
    width=WALL_THICKNESS,
    height=ROOM_HEIGHT,
    depth=ROOM_LENGTH
)

wall_left.translate([
    0,
    0,
    0
])


# ----------------------------------------------------------
# Right side wall
# ----------------------------------------------------------

wall_right = o3d.geometry.TriangleMesh.create_box(
    width=WALL_THICKNESS,
    height=ROOM_HEIGHT,
    depth=ROOM_LENGTH
)

wall_right.translate([
    ROOM_WIDTH - WALL_THICKNESS,
    0,
    0
])


# ----------------------------------------------------------
# Combine environment
# ----------------------------------------------------------

env = (
    floor
    + wall_left
    + wall_right
)

env.compute_vertex_normals()


# ==========================================================
# RAYCASTING SCENE
# ==========================================================

tmesh = o3d.t.geometry.TriangleMesh.from_legacy(env)

scene = o3d.t.geometry.RaycastingScene()

scene.add_triangles(tmesh)


# ==========================================================
# 3D LiDAR
# ==========================================================

HFOV = 360.0

VUPFOV = 70.0
VDOWNFOV = 70.0

NUM_H = 256
NUM_V = 128


# ----------------------------------------------------------
# Vertical angles
# ----------------------------------------------------------

v_angles = np.linspace(
    -VDOWNFOV,
    VUPFOV,
    NUM_V
)


# ----------------------------------------------------------
# Horizontal angles
# ----------------------------------------------------------

h_angles = np.linspace(
    0,
    HFOV,
    NUM_H,
    endpoint=False
)


# ----------------------------------------------------------
# Generate ray directions
# ----------------------------------------------------------

dirs = []

for v_deg in v_angles:

    v = np.deg2rad(v_deg)

    for h_deg in h_angles:

        h = np.deg2rad(h_deg)

        x = np.cos(v) * np.cos(h)
        y = np.sin(v)
        z = np.cos(v) * np.sin(h)

        dirs.append([
            x,
            y,
            z
        ])


dirs = np.asarray(
    dirs,
    dtype=np.float32
)

print("Total rays:", len(dirs))


# ==========================================================
# GENERATE FORWARD SCAN POSITIONS
# ==========================================================

# Center of corridor
center_x = ROOM_WIDTH / 2.0

# Apply lateral offset
lidar_x = center_x + LATERAL_OFFSET

# Make sure LiDAR does not start inside a wall
if lidar_x <= WALL_THICKNESS:
    raise ValueError(
        "LATERAL_OFFSET places LiDAR inside the left wall."
    )

if lidar_x >= ROOM_WIDTH - WALL_THICKNESS:
    raise ValueError(
        "LATERAL_OFFSET places LiDAR inside the right wall."
    )


# Generate Z positions
z_positions = np.arange(
    START_Z,
    END_Z + FORWARD_INCREMENT,
    FORWARD_INCREMENT
)


scan_positions = [
    np.array([
        lidar_x,
        LIDAR_HEIGHT,
        z
    ], dtype=np.float32)

    for z in z_positions
]


print("Number of frames:", len(scan_positions))
print("Start position:", scan_positions[0])
print("End position:", scan_positions[-1])
print("Forward increment:", FORWARD_INCREMENT)
print("Lateral offset:", LATERAL_OFFSET)


# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

os.makedirs(
    "datasetLong/scans",
    exist_ok=True
)


# ==========================================================
# POSE FILE
# ==========================================================

pose_file = open(
    "datasetLong/poses.csv",
    "w"
)

pose_file.write(
    "frame,timestamp,x,y,z\n"
)


# ==========================================================
# LiDAR VISUALIZATION OBJECT
# ==========================================================

origin = scan_positions[0]

lidar = o3d.geometry.TriangleMesh.create_sphere(
    radius=0.3
)

lidar.compute_vertex_normals()

lidar.paint_uniform_color([
    1,
    0,
    1
])

lidar.translate(origin)


# ==========================================================
# FIRST SCAN
# ==========================================================

hit_points = lidar_scan(
    scene,
    origin,
    dirs
)

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    hit_points
)

pcd.paint_uniform_color([
    1,
    1,
    0
])


# ==========================================================
# AXIS
# ==========================================================

points = np.array([
    [0, 0, 0],
    [10, 0, 0],
    [0, 10, 0],
    [0, 0, 10]
])

lines = np.array([
    [0, 1],
    [0, 2],
    [0, 3]
])

colors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

line = o3d.geometry.LineSet()

line.points = o3d.utility.Vector3dVector(
    points
)

line.lines = o3d.utility.Vector2iVector(
    lines
)

line.colors = o3d.utility.Vector3dVector(
    colors
)


# ==========================================================
# VISUALIZATION
# ==========================================================

vis = o3d.visualization.Visualizer()

vis.create_window(
    window_name="Long Dataset LiDAR"
)

vis.add_geometry(env)
vis.add_geometry(line)
vis.add_geometry(pcd)
vis.add_geometry(lidar)


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
    ROOM_WIDTH / 2,
    ROOM_HEIGHT / 2,
    ROOM_LENGTH / 2
])

ctr.set_zoom(
    0.15
)


# ==========================================================
# SCAN LOOP
# ==========================================================

for frame_id, origin in enumerate(scan_positions):

    # ------------------------------------------------------
    # Move LiDAR
    # ------------------------------------------------------

    current_center = np.asarray(
        lidar.get_center()
    )

    lidar.translate(
        origin - current_center
    )


    # ------------------------------------------------------
    # Raycast
    # ------------------------------------------------------

    hit_points = lidar_scan(
        scene,
        origin,
        dirs
    )


    # ------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------

    timestamp = time.time()


    # ------------------------------------------------------
    # Save scan
    # ------------------------------------------------------

    scan_filename = (
        f"datasetLong/scans/"
        f"scan_{frame_id:06d}.csv"
    )


    header = (
        f"# timestamp={timestamp}\n"
        f"# sensor_x={origin[0]}\n"
        f"# sensor_y={origin[1]}\n"
        f"# sensor_z={origin[2]}\n"
        f"x,y,z"
    )


    np.savetxt(
        scan_filename,
        hit_points,
        delimiter=",",
        header=header,
        comments=""
    )


    # ------------------------------------------------------
    # Save pose
    # ------------------------------------------------------

    pose_file.write(
        f"{frame_id},"
        f"{timestamp},"
        f"{origin[0]},"
        f"{origin[1]},"
        f"{origin[2]}\n"
    )


    # ------------------------------------------------------
    # Update visualization
    # ------------------------------------------------------

    pcd.points = o3d.utility.Vector3dVector(
        hit_points
    )

    vis.update_geometry(lidar)
    vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()


    # ------------------------------------------------------
    # Small delay
    # ------------------------------------------------------

    time.sleep(0.5)


# ==========================================================
# FINISH
# ==========================================================

pose_file.close()

print()
print("========================================")
print("Finished long dataset generation.")
print("========================================")
print(
    "Dataset : datasetLong/"
)
print(
    "Frames  :", len(scan_positions)
)
print(
    "Length  :", ROOM_LENGTH, "m"
)
print(
    "Width   :", ROOM_WIDTH, "m"
)
print(
    "Increment:", FORWARD_INCREMENT, "m"
)
print(
    "Offset  :", LATERAL_OFFSET, "m"
)

vis.run()

vis.destroy_window()