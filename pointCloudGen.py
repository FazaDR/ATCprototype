import open3d as o3d
import numpy as np
import time
import os


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

points = np.array([
    [0, 0, 0],
    [10, 0, 0],
    [0, 10, 0],
    [0, 0, 10]
])

lines = np.array([
    [0,1],
    [0,2],
    [0,3]
])

colors = np.array([
    [1, 0, 0],  # X = red
    [0, 1, 0],  # Y = green
    [0, 0, 1]   # Z = blue
])


line = o3d.geometry.LineSet()
line.points = o3d.utility.Vector3dVector(points)
line.lines = o3d.utility.Vector2iVector(lines)
line.colors = o3d.utility.Vector3dVector(colors)


ROOM_SIZE = 8.0
ROOM_HEIGHT = 3.0
WALL_THICKNESS = 0.2


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

wall1.translate([0,0,0])
wall2.translate([ROOM_SIZE,0,0])
wall3.translate([0,0,ROOM_SIZE])
wall4.translate([0,0,0])

divider = o3d.geometry.TriangleMesh.create_box(
    width=0.15,
    height=ROOM_HEIGHT,
    depth=5.0
)

divider.translate([
    ROOM_SIZE/2 - 0.075,
    0,
    1.5
])

env = floor + divider + wall1 + wall2 + wall3 + wall4

env.compute_vertex_normals()


# ==========================================================
# 3D LiDAR
# ==========================================================

tmesh = o3d.t.geometry.TriangleMesh.from_legacy(env)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(tmesh)

# lidar position
scan_positions = [
    np.array([1.50, 2.0, 1.50], dtype=np.float32),
    np.array([3.17, 2.0, 1.50], dtype=np.float32),
    np.array([4.83, 2.0, 1.50], dtype=np.float32),

    np.array([6.50, 2.0, 1.50], dtype=np.float32),
    np.array([6.50, 2.0, 3.17], dtype=np.float32),
    np.array([6.50, 2.0, 4.83], dtype=np.float32),

    np.array([6.50, 2.0, 6.50], dtype=np.float32),
    np.array([4.83, 2.0, 6.50], dtype=np.float32),
    np.array([3.17, 2.0, 6.50], dtype=np.float32),

    np.array([1.50, 2.0, 6.50], dtype=np.float32),
    np.array([1.50, 2.0, 4.83], dtype=np.float32),
    np.array([1.50, 2.0, 3.17], dtype=np.float32),
]

origin = scan_positions[0].copy()

lidar = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
lidar.compute_vertex_normals()
lidar.paint_uniform_color([1, 0, 1])

lidar.translate(origin)

# ==========================================================
# LiDAR specification
# ==========================================================

HFOV = 360.0          # degrees
VUPFOV = 70.0           # degrees
VDOWNFOV = 70.0         # degrees

NUM_H = 128          # horizontal rays
NUM_V = 64            # vertical channels

# vertical angles
v_angles = np.linspace(
    -VDOWNFOV,
    VUPFOV,
    NUM_V
)

# horizontal angles
h_angles = np.linspace(
    0,
    HFOV,
    NUM_H,
    endpoint=False
)

dirs = []

for v_deg in v_angles:

    v = np.deg2rad(v_deg)

    for h_deg in h_angles:

        h = np.deg2rad(h_deg)

        x = np.cos(v) * np.cos(h)
        y = np.sin(v)
        z = np.cos(v) * np.sin(h)

        dirs.append([x, y, z])

dirs = np.asarray(
    dirs,
    dtype=np.float32
)

print("Total rays:", len(dirs))



hit_points = lidar_scan(scene, origin, dirs)

# create point cloud from lidar returns
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(hit_points)

# optional: make lidar points yellow
pcd.paint_uniform_color([1, 1, 0])

# ==========================================================
# Dataset output
# ==========================================================

os.makedirs("dataset/scans", exist_ok=True)

pose_file = open(
    "dataset/poses.csv",
    "w"
)

pose_file.write(
    "frame,timestamp,x,y,z\n"
)

frame_id = 0

# ==========================================================
# Visualize
# ==========================================================

vis = o3d.visualization.Visualizer()
vis.create_window()


vis.add_geometry(env)
vis.add_geometry(line)
vis.add_geometry(pcd)
vis.add_geometry(lidar)

ctr = vis.get_view_control()

ctr.set_front([0.0, 1.0, 0.0])
ctr.set_up([0.0, 0.0, 1.0])
ctr.set_lookat([ROOM_SIZE/2, ROOM_HEIGHT/2, ROOM_SIZE/2])
ctr.set_zoom(0.35)

for origin in scan_positions:

    lidar.translate(
        origin - np.asarray(lidar.get_center())
    )

    hit_points = lidar_scan(scene, origin, dirs)

    timestamp = time.time()

    scan_filename = (
        f"dataset/scans/scan_{frame_id:06d}.csv"
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

    pose_file.write(
        f"{frame_id},"
        f"{timestamp},"
        f"{origin[0]},"
        f"{origin[1]},"
        f"{origin[2]}\n"
    )

    frame_id += 1

    pcd.points = o3d.utility.Vector3dVector(hit_points)

    vis.update_geometry(lidar)
    vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    time.sleep(1)

print("Finished scanning.")

vis.run()
vis.destroy_window()

pose_file.close()