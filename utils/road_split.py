import os
import numpy as np
import open3d as o3d
from utils.common_utils import pc_numpy_2_o3d


def load_road_split_labels(label_path):
    """
    Load road split labels from a binary file.

    :param label_path: Path to the binary file containing the labels
    :return: NumPy array of shape (N, 1) containing the road split labels
    """
    labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
    return labels


def split_pc(labels):
    """
    Split point cloud indices based on semantic labels.

    :param labels: NumPy array of shape (N, 1) containing semantic labels for each point
    :return: Tuple containing four lists of indices:
             - inx_road_arr: Indices of points labeled as road (40)
             - inx_other_road_arr: Indices of points labeled as parking (44) or sidewalk (48)
             - inx_other_ground_arr: Indices of points labeled as terrain (70 or 71)
             - inx_no_road_arr: Indices of points not matching any of the above categories
    """
    inx_road_arr = []   
    inx_other_road_arr = [] 
    inx_other_ground_arr = []   
    inx_no_road_arr = []    
    '''
        40: "road"          
        44: "parking"       
        48: "sidewalk"      
        49: "other-ground"  
        72: "terrain"       
    '''
    for i in range(len(labels)):
        lb = labels[i][0]
        if lb == 40:
            inx_road_arr.append(i)  
        elif lb == 44:
            inx_other_road_arr.append(i)
        elif lb == 48:
            inx_other_road_arr.append(i)    
        elif lb in (70, 71):
            inx_other_ground_arr.append(i) 
        else:
            inx_no_road_arr.append(i)   
    return inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr


def road_split(pc, road_pc_path, road_label_path):
    """
    Split point cloud into road and non-road points using semantic labels.

    This function either loads a pre-computed road point cloud or generates a new one
    by filtering and processing the input point cloud based on semantic labels.
    It creates a clean road surface using alpha shapes and outlier removal techniques.

    :param pc: Input point cloud as a NumPy array of shape (N, 3)
    :param road_pc_path: Path to save/load the processed road point cloud
    :param road_label_path: Path to the semantic labels file
    :return: Tuple containing:
             - road_pc: Processed road point cloud as NumPy array
             - _pc_non_road: Non-road points from the input point cloud
             - labels: Semantic labels loaded from file
    """
    pc_path = road_pc_path
    label_path = road_label_path

    if os.path.exists(pc_path):
        labels = load_road_split_labels(label_path)
        road_pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 3))
        inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = split_pc(labels)

        _pc_non_road = pc[inx_other_road_arr + inx_other_ground_arr + inx_no_road_arr]
    else:
        labels = load_road_split_labels(label_path)

        inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = split_pc(labels)
        if len(inx_road_arr) <= 10:
            return None, None, None, None

        _pc_road, _pc_other_road, _pc_other_ground, _pc_no_road = \
            pc[inx_road_arr], pc[inx_other_road_arr], pc[inx_other_ground_arr], pc[inx_no_road_arr]

        _pc_non_road = pc[inx_other_road_arr + inx_other_ground_arr + inx_no_road_arr]
                
        pcd_road = pc_numpy_2_o3d(_pc_road)

        cl, ind = pcd_road.remove_radius_outlier(nb_points=7, radius=1)
        pcd_inlier_road = pcd_road.select_by_index(ind)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_inlier_road, 10)

        pcd_inter = mesh.sample_points_uniformly(number_of_points=50000)

        _pc_inter = np.asarray(pcd_inter.points)
        dis = np.linalg.norm(_pc_inter, axis=1, ord=2)
        _pc_inter_valid = _pc_inter[dis > 4]
        
        road_pc = _pc_inter_valid.astype(np.float32)
        road_pc.astype(np.float32).tofile(pc_path, )

    return road_pc, _pc_non_road, labels


if __name__ == '__main__':
    ...
