import re
import yaml
import os
import math
import numpy as np
import open3d as o3d


def load_yaml(file, opt=None):
    """
    Load a V2X4Real YAML configuration file and return it as a dictionary.

    :param file: Yaml file path.
    :param opt: Argparser.
    :return param: A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param


def get_labels(rz_degree, lidar_box, image_box, truncation_ratio):
    """
    Convert V2X box data to KITTI format labels.

    :param rz_degree: Rotation angle around Z-axis (yaw) in degrees
    :param lidar_box: 3D bounding box object from LiDAR detection
    :param image_box: Optional 2D bounding box in image coordinates (xmin, ymin, xmax, ymax)
    :param truncation_ratio: Optional truncation ratio of the object in the image
    :return: List of strings representing the KITTI formatted label line
    """
    place_holder = -1111
    label_2_prefix = ["Car", "0.00", "0", "-10000"]

    img_xmin, img_ymin, img_xmax, img_ymax = place_holder, place_holder, place_holder, place_holder
    if image_box is not None:
        img_xmin, img_ymin, img_xmax, img_ymax = image_box
    if truncation_ratio is not None:
        label_2_prefix[1] = str(round(truncation_ratio, 2))

    x_, y_, z_ = np.asarray(lidar_box.extent)
    h, w, l = z_, y_, x_

    corners = np.asarray(lidar_box.get_box_points())
    x_min, y_min, z_min = np.min(corners, axis=0)
    x_max, y_max, z_max = np.max(corners, axis=0)
    lidar_bottom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]).reshape((1, 3))

    x, y, z = lidar_bottom_center[0, 0], lidar_bottom_center[0, 1], lidar_bottom_center[0, 2]

    r_y = math.radians(rz_degree) + np.pi / 2
    paras = [img_xmin, img_ymin, img_xmax, img_ymax, h, w, l, x, y, z, r_y]
    label_2_suffix = [str(round(para, 2)) for para in paras]
    label_2_prefix.extend(label_2_suffix)
    return label_2_prefix


def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    :param data: The dictionary contains all data.
    :param save_name: Full path of the output yaml file.
    """

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def read_Bin_PC(path, retrun_r=False):
    """
    Read a point cloud from a binary file or NumPy array file.

    :param path: Path to the point cloud file (.bin or .npy)
    :param retrun_r: If True, return the full point data including the fourth channel.
                     If False, return only the XYZ coordinates (default: False)
    """
    if path.split(".")[-1] == "npy":
        example = np.load(path).astype(np.float32)
    else:
        example = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    example_xyz = example[:, :3]

    if retrun_r:
        return example
    return example_xyz


def pcd_to_np(pcd_file):
    """
    Read  pcd and return numpy array.

    :param pcd_file: The pcd file that contains the point cloud.
    :return pcd: PointCloud object, used for visualization
            pcd_np: The lidar data in numpy format, shape:(n, 4)
    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)

    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def complet_pc(mixed_pc_three):
    """
    Complete a 3-channel point cloud by adding a fourth channel of zeros.

    :param mixed_pc_three: A numpy array of shape (N, 3) representing point cloud data with XYZ coordinates
    """
    assert mixed_pc_three.shape[1] == 3

    hang = mixed_pc_three.shape[0]
    b = np.zeros((hang, 1))
    mixed_pc = np.concatenate([mixed_pc_three, b], axis=1)
    return mixed_pc


def load_pc(bg_index, pc_path):
    """
    Load a background point cloud from a binary file.

    :param bg_index: Index of the background frame to load
    :param pc_path: Directory containing the point cloud files
    """
    bg_pc_path = os.path.join(pc_path, f"{bg_index:06d}.bin")
    bg_xyz = read_Bin_PC(bg_pc_path)
    return bg_xyz


def load_label(bg_index, label_path):
    """
    Load the label data for a specific background frame.

    :param bg_index: Index of the background frame to load
    :param label_path: Directory containing the label files
    """
    bg_label_path = os.path.join(label_path, f"{bg_index:06d}.yaml")
    param = load_yaml(bg_label_path)
    return param
