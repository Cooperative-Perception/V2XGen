import config
import math
import copy
import open3d as o3d
import numpy as np
from shapely.geometry import Polygon
from build.mtest.utils import box_utils
from scipy.spatial.transform import Rotation as RR


def pc_numpy_2_o3d(pc):
    """
    Convert a NumPy-formatted point cloud to an Open3D point cloud object.

    Supports two input formats:
    1. 4-channel point cloud (Nx4): contains x, y, z coordinates and intensity
    2. 3-channel point cloud (Nx3): contains only x, y, z coordinates

    :param pc: Point cloud data as a NumPy array, shape (N, 3) or (N, 4)
    :return: Open3D PointCloud object
    """
    if isinstance(pc, np.ndarray) and pc.shape[1] == 4:
        xyz = pc[:, :3]
        intensity = pc[:, 3:4]
        colors = np.hstack((intensity, intensity, intensity))  # intensity flat to color
        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(xyz)
        pcd_bg.colors = o3d.utility.Vector3dVector(colors)
        return pcd_bg
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(pc)
    return pcd_bg


def pcd_to_np(pcd):
    """
    convert pcd to numpy array.
    """
    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)

    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def get_obj_pcd_from_corner(corner, bg_pcd):
    """
    Convert Open3D PointCloud object to NumPy array.

    :param corner:
    :param bg_pcd: Open3D PointCloud object
    :return: NumPy array of shape (N, 4) with dtype float32
    """
    min_bound = np.min(corner, axis=0)
    max_bound = np.max(corner, axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    obj_pcd = bg_pcd.crop(bbox)
    return obj_pcd


def get_corners(param):
    """
    Compute the 3D bounding box corners for all vehicles in the scene.

    Processes vehicle parameters from a scene dictionary, converts them into
    LiDAR coordinate system bounding boxes, and computes the 3D corners for each box.

    :param param: Dictionary containing scene information, must have a 'vehicles' key
                  with vehicle details (angle, extent, location, center)
    :return: Array of shape (N, 8, 3) representing the 3D corners of each vehicle's bounding box
    """
    objs_ry = []
    objs_l = []
    objs_w = []
    objs_h = []
    objs_loc = []

    for i, vehicle in param["vehicles"].items():
        objs_ry.insert(i, vehicle['angle'][1] * np.pi / 180)
        objs_l.insert(i, vehicle['extent'][0] * 2)
        objs_w.insert(i, vehicle['extent'][1] * 2)
        objs_h.insert(i, vehicle['extent'][2] * 2)
        arr_loc = np.array(vehicle['location']) + np.array(vehicle['center'])
        objs_loc.insert(i, arr_loc)
    rots = np.array(objs_ry)

    l, h, w = np.array(objs_l).reshape(-1, 1), np.array(objs_h).reshape(-1, 1), np.array(objs_w).reshape(-1, 1)
    loc_lidar = np.array(objs_loc)

    rots = rots[..., np.newaxis]

    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots], axis=1)

    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)

    return corners_lidar


def get_corners_positions(param):
    """
    Compute 3D bounding box corners and return vehicle position information.

    :param param: Dictionary containing scene information, must have a 'vehicles' key
                 with vehicle details (angle, extent, location, center)
    :return: Tuple containing:
            - corners_lidar: Array of shape (N, 8, 3) representing the 3D corners of each bounding box
            - objs_ry_degree: List of yaw angles in degrees for each vehicle
            - objs_loc: List of vehicle locations in LiDAR coordinates
            - objs_h: List of vehicle heights
    """
    objs_ry = []
    objs_ry_degree = []
    objs_l = []
    objs_w = []
    objs_h = []
    objs_loc = []

    for i, vehicle in param["vehicles"].items():
        objs_ry.insert(i, vehicle['angle'][1] * np.pi / 180)
        objs_ry_degree.insert(i, vehicle['angle'][1])
        objs_l.insert(i, vehicle['extent'][0] * 2)
        objs_w.insert(i, vehicle['extent'][1] * 2)
        objs_h.insert(i, vehicle['extent'][2] * 2)
        arr_loc = np.array(vehicle['location']) + np.array(vehicle['center'])
        objs_loc.insert(i, arr_loc)
    rots = np.array(objs_ry)

    l, h, w = np.array(objs_l).reshape(-1, 1), np.array(objs_h).reshape(-1, 1), np.array(objs_w).reshape(-1, 1)
    loc_lidar = np.array(objs_loc)

    rots = rots[..., np.newaxis]
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots], axis=1)

    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)

    return corners_lidar, objs_ry_degree, objs_loc, objs_h


def find_cp_vehicle_id(cp_info, ego_id):
    """
    Find the cooperative perception vehicle ID corresponding to the ego vehicle ID.

    :param cp_info: Cooperative perception information object containing vehicle data
    :param ego_id: ID of the ego vehicle to find in the cooperative perception data
    :return: Tuple containing:
             - cp_id: The cooperative perception vehicle ID if found, otherwise -1
             - found: Boolean indicating whether a match was found
    """
    for cp_id, cp_vehicle in cp_info.param['vehicles'].items():
        if cp_vehicle['ass_id'] == ego_id:
            return cp_id, True

    return -1, False


def get_pc_index_in_corner(pc, corner):
    """
    Find the indices of point cloud points that lie within a 3D bounding box defined by its corners.

    :param pc: Input point cloud as a NumPy array of shape (N, 3) or (N, 4)
    :param corner: Array of shape (8, 3) representing the 8 corners of a 3D bounding box
    :return: NumPy array of indices of points that lie within the bounding box
    """
    min_bound = np.min(corner, axis=0)
    max_bound = np.max(corner, axis=0)
    indices = np.where(
        (min_bound[0] <= pc[:, 0]) & (pc[:, 0] <= max_bound[0]) &
        (min_bound[1] <= pc[:, 1]) & (pc[:, 1] <= max_bound[1]) &
        (min_bound[2] <= pc[:, 2]) & (pc[:, 2] <= max_bound[2])
    )[0]

    return indices


def center_system_transform(center, cur_lidar_pose, tar_lidar_pose):
    """
    Transform a 3D point from one LiDAR coordinate system to another.

    :param center: 3D point coordinates in the current LiDAR frame (shape: [3])
    :param cur_lidar_pose: 4x4 transformation matrix of the current LiDAR in world coordinates
    :param tar_lidar_pose: 4x4 transformation matrix of the target LiDAR in world coordinates
    :return: Transformed 3D point coordinates in the target LiDAR frame (shape: [3])
    """
    center_pos = np.append(center, 1)
    T_cur_tar = np.linalg.inv(tar_lidar_pose) @ cur_lidar_pose
    target_pos = T_cur_tar @ center_pos
    return target_pos[:3]


def points_system_transform(points, cur_lidar_pose, tar_lidar_pose):
    """
    Transform a set of 3D points from one LiDAR coordinate system to another.

    :param points: List or array of 3D points in the current LiDAR frame (shape: [N, 3])
    :param cur_lidar_pose: 4x4 transformation matrix of the current LiDAR in world coordinates
    :param tar_lidar_pose: 4x4 transformation matrix of the target LiDAR in world coordinates
    :return: List of transformed 3D points in the target LiDAR frame (shape: [N, 3])
    """
    tar_points = []
    for pt in points:
        tar_pt = center_system_transform(pt, cur_lidar_pose, tar_lidar_pose)
        tar_points.append(tar_pt)
    return tar_points


def rz_degree_system_transform(rz_degree, cur_lidar_pose, tar_lidar_pose):
    """
    Transform a rotation angle around the Z-axis from one LiDAR coordinate system to another.

    :param rz_degree: Rotation angle around Z-axis in degrees (yaw) in current LiDAR frame
    :param cur_lidar_pose: 4x4 transformation matrix of the current LiDAR in world coordinates
    :param tar_lidar_pose: 4x4 transformation matrix of the target LiDAR in world coordinates
    :return: Rotation angle around Z-axis in degrees in target LiDAR frame
    """
    theta = np.radians(rz_degree)

    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    T = tar_lidar_pose @ np.linalg.inv(cur_lidar_pose)
    T_inv = np.linalg.inv(T)

    Rz_tar = T @ Rz @ T_inv

    theta_tar = np.arctan2(Rz_tar[1, 0], Rz_tar[0, 0])

    rz_degree_tar = np.degrees(theta_tar)

    return rz_degree_tar


def is_box_containing_points(box, points):
    """
    Check if a 3D bounding box contains any of the given points.

    :param box: Array of shape (8, 3) representing the 8 corners of a 3D bounding box
    :param points: List or array of 3D points (shape: [N, 3]) to check against the box
    :return: True if any point is inside the box, False otherwise
    """
    min_bound = np.min(box, axis=0)
    max_bound = np.max(box, axis=0)

    for pt in points:
        if (min_bound[0] <= pt[0] <= max_bound[0] and
                min_bound[1] <= pt[1] <= max_bound[1] and
                min_bound[2] <= pt[2] <= max_bound[2]):
            return True
        else:
            continue

    return False


def is_box_containing_position(box, position):
    """
    Check if a single 3D position lies within a 3D bounding box.

    :param box: Array of shape (8, 3) representing the 8 corners of a 3D bounding box
    :param position: 3D point coordinates (shape: [3]) to check against the box
    :return: True if the position is inside the box, False otherwise
    """
    min_bound = np.min(box, axis=0)
    max_bound = np.max(box, axis=0)

    if (min_bound[0] <= position[0] <= max_bound[0] and
            min_bound[1] <= position[1] <= max_bound[1] and
            min_bound[2] <= position[2] <= max_bound[2]):
        return True

    return False


def is_boxes_overlap(box1, box2):
    """
    Check if two 3D bounding boxes overlap in their 2D projections.

    :param box1: Array of shape (8, 3) representing the 8 corners of the first 3D bounding box
    :param box2: Array of shape (8, 3) representing the 8 corners of the second 3D bounding box
    :return: True if the 2D projections of the boxes intersect, False otherwise
    """
    box1_2d = box1[:, :2]
    box2_2d = box2[:, :2]

    poly1 = Polygon(box1_2d)
    poly2 = Polygon(box2_2d)

    return poly1.intersects(poly2)


def param_vehicle_check(param):
    """
    Print the IDs of all vehicles in the parameter dictionary.

    :param param: Dictionary containing vehicle information with a 'vehicles' key
    """
    for key, value in param['vehicles'].items():
        print(key)


def get_geometric_info(obj):
    """
    Calculate geometric properties of a 3D object.

    :param obj: Object with get_min_bound() and get_max_bound() methods
                (e.g., Open3D geometry object)
    :return: Tuple containing:
             - half_diagonal: Half the length of the diagonal in the XY-plane
             - center: Center coordinates [x, y, z]
             - half_height: Half the height of the object
    """
    min_xyz = obj.get_min_bound()
    max_xyz = obj.get_max_bound()
    x_min, x_max = min_xyz[0], max_xyz[0]
    y_min, y_max = min_xyz[1], max_xyz[1]
    z_min, z_max = min_xyz[2], max_xyz[2]
    half_diagonal = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 2
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    half_height = (max_xyz[2] - min_xyz[2]) / 2

    return half_diagonal, center, half_height


def corner_to_line_set_box(corner, line_color=[1, 0, 0]):
    """
    Create an Open3D LineSet representing a 3D bounding box from its corners.

    The corners should be ordered as follows:
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2

    :param corner: Array of shape (8, 3) representing the 8 corners of the 3D bounding box
    :param line_color: RGB color for the box lines (default: red [1, 0, 0])
    :return: Open3D LineSet object representing the 3D bounding box
    """
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7],
                          [4, 5], [5, 6], [6, 7], [7, 4]])

    colors = np.array([line_color for _ in range(len(lines_box))])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner)
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_cylinder_between_points(point1, point2, radius=0.02, resolution=20):
    """
    Create a 3D cylinder between two points in space.

    :param point1: First endpoint of the cylinder (shape: [3])
    :param point2: Second endpoint of the cylinder (shape: [3])
    :param radius: Radius of the cylinder (default: 0.02)
    :param resolution: Number of vertices around the circumference (default: 20)
    :return: Open3D TriangleMesh object representing the cylinder
    """
    point1 = np.array(point1)
    point2 = np.array(point2)

    height = np.linalg.norm(point2 - point1)

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    cylinder.compute_vertex_normals()

    midpoint = (point2 + point1) / 2
    direction = (point2 - point1) / height

    cylinder.translate(midpoint)

    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    angle = np.arccos(np.dot(z_axis, direction))
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(rotation_matrix, center=midpoint)

    return cylinder


def get_box_corners(center, extent):
    """
    Calculate the 8 corners of an axis-aligned 3D bounding box.

    :param center: Center coordinates of the box [x, y, z]
    :param extent: Half-extents of the box [l, w, h] representing half of
                   the length, width, and height respectively
    :return: NumPy array of shape (8, 3) containing the coordinates of all 8 corners
    """
    x, y, z = center
    l, w, h = extent

    corners = [
        [x + l, y + w, z + h],
        [x + l, y + w, z - h],
        [x + l, y - w, z + h],
        [x + l, y + w, z - h],
        [x - l, y + w, z + h],
        [x - l, y + w, z - h],
        [x - l, y - w, z + h],
        [x - l, y - w, z - h],
    ]

    return np.array(corners)


def get_euler_from_matrix(R):
    """
    Convert a rotation matrix to Euler angles.

    :param R: 3x3 rotation matrix (numpy array)
    :return: Tuple containing the Euler angles in radians for rotations around:
             - X-axis (sciangle_0)
             - Y-axis (sciangle_1)
             - Z-axis (sciangle_2)
    """
    euler_type = "XYZ"

    sciangle_0, sciangle_1, sciangle_2 = RR.from_matrix(R).as_euler(euler_type)

    return sciangle_0, sciangle_1, sciangle_2


def get_box3d_R(box3d):
    """
    Get a copy of the rotation matrix from a 3D box object.

    :param box3d: A 3D box object that contains a rotation matrix (R) as an attribute
    :return: A copy of the 3x3 rotation matrix from the box3d object
    """
    return copy.copy(box3d.R)


def crop_point_cloud(pc, center, size):
    """
    Crop a point cloud to a square region in the XY-plane.

    :param pc: Input point cloud as a NumPy array of shape (N, 3) or (N, 4)
    :param center: Center coordinates of the cropping region [x, y, z]
    :param size: Side length of the square cropping region
    :return: Cropped point cloud as a NumPy array
    """
    x, y, z = center

    x_min = x - size / 2
    x_max = x + size / 2
    y_min = y - size / 2
    y_max = y + size / 2

    cropped_pc = pc[
        (pc[:, 0] >= x_min) & (pc[:, 0] <= x_max) &
        (pc[:, 1] >= y_min) & (pc[:, 1] <= y_max)
        ]

    return cropped_pc

