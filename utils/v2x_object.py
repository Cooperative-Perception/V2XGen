import os
import sys
import yaml
import numpy as np
import open3d as o3d
from utils.v2X_file import read_Bin_PC, load_yaml, pcd_to_np
from utils.road_split import road_split
from build.mtest.utils import box_utils
from utils.common_utils import pc_numpy_2_o3d


class V2XInfo:
    """
    Load and manage V2X (Vehicle-to-Everything) data, including vehicle info, road point clouds, and labels.
    """

    def __init__(self, bg_index, is_ego=True, dataset_config=None):
        """
        Initialize the V2XInfo class by loading data and initializing core attributes.

        :param bg_index: Index of the background frame to load (for matching point cloud/label files).
        :param is_ego: Boolean indicating if the data belongs to the ego vehicle (True) or a cooperative vehicle (False).
        :param dataset_config: Configuration object containing paths to dataset directories (e.g., ego/coop PC/label dirs).
        """
        self.bg_index = bg_index
        self.is_ego = is_ego
        self.dataset_config = dataset_config

        path_info = self.load_data_path()

        # read pcd data
        bg_pc = pcd_to_np(path_info['bg_path'])         # read background pcd from .pcd files
        road_pc, non_road_pc, road_label = road_split(bg_pc[:, :3], path_info['road_path'],
                                                      path_info['road_label_path'])

        # read bin data
        # bg_xyz = read_Bin_PC(path_info['bg_path'])  # read bg pcd from .bin files
        # road_pc, non_road_pc, road_label = road_split(bg_xyz, path_info['road_path'],
        #                                               path_info['road_label_path'])

        param = load_yaml(path_info['param_path'])

        self.pc = bg_pc
        self.param = param
        self.road_pc = road_pc
        self.no_road_pc = non_road_pc
        self.road_label = road_label
        self.vehicles_info = {}
        self.recent_deleted_car_id = -1

        self.load_vehicles_info()

    def get_vehicles_nums(self):
        return len(list(self.param['vehicles'].keys()))

    def load_vehicles_info(self):
        """
        Parse vehicle parameters from self.param and store in self.vehicles_info.

        Extracts vehicle pose (yaw, location), dimensions (length/width/height),
        and computes 3D bounding box corners for each vehicle.
        """
        objs_yaw = []
        objs_yaw_degree = []
        objs_l = []
        objs_w = []
        objs_h = []
        objs_loc = []
        for i, vehicle in self.param["vehicles"].items():
            yaw_degree = vehicle['angle'][1]
            length = vehicle['extent'][0] * 2
            width = vehicle['extent'][1] * 2
            height = vehicle['extent'][2] * 2
            center = np.array(vehicle['location']) + np.array(vehicle['center'])

            objs_yaw.append(yaw_degree * np.pi / 180)
            objs_yaw_degree.append(yaw_degree)
            objs_l.append(length)
            objs_w.append(width)
            objs_h.append(height)
            objs_loc.append(center)

            self.vehicles_info[i] = {
                'yaw_degree': yaw_degree,
                'length': length,
                'width': width,
                'height': height,
                'center': center
            }
        rots = np.array(objs_yaw)

        l, h, w = np.array(objs_l).reshape(-1, 1), np.array(objs_h).reshape(-1, 1), np.array(objs_w).reshape(-1, 1)
        loc_lidar = np.array(objs_loc)

        rots = rots[..., np.newaxis]
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots], axis=1)
        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar).tolist()

        for i in self.vehicles_info.keys():
            self.vehicles_info[i]["corner"] = corners_lidar[0]
            corners_lidar = corners_lidar[1:]

    def load_data_path(self):
        """
        Construct paths to point cloud, label, and road data files.

        :return: Dictionary containing paths to background PC, param YAML, road PC, and road label.
        """
        # dataset path config
        # bg_pc_path = os.path.join(self.dataset_config.ego_pc_dir, f"{self.bg_index:06d}.bin")
        bg_pc_path = os.path.join(self.dataset_config.ego_pc_dir, f"{self.bg_index:06d}.pcd")
        bg_yaml_path = os.path.join(self.dataset_config.ego_label_dir, f"{self.bg_index:06d}.yaml")
        road_split_pc_dir = self.dataset_config.ego_road_split_pc_dir
        road_split_label_dir = self.dataset_config.ego_road_split_label_dir
        if not self.is_ego:
            # bg_pc_path = os.path.join(self.dataset_config.coop_pc_dir, f"{self.bg_index:06d}.bin")
            bg_pc_path = os.path.join(self.dataset_config.coop_pc_dir, f"{self.bg_index:06d}.pcd")
            bg_yaml_path = os.path.join(self.dataset_config.coop_label_dir, f"{self.bg_index:06d}.yaml")
            road_split_pc_dir = self.dataset_config.coop_road_split_pc_dir
            road_split_label_dir = self.dataset_config.coop_road_split_label_dir

        os.makedirs(road_split_pc_dir, exist_ok=True)

        road_pc_path = f"{road_split_pc_dir}/{self.bg_index:06d}.bin"
        road_label_path = f"{road_split_label_dir}/{self.bg_index:06d}.label"

        path_info = {
            'bg_path': bg_pc_path,
            'param_path': bg_yaml_path,
            'road_path': road_pc_path,
            'road_label_path': road_label_path
        }

        return path_info

    def delete_vehicle_of_id(self, car_id):
        """
        Delete a vehicle from the current frame by ID.

        :param car_id: ID of the vehicle to delete (must exist in self.param['vehicles']).
        """
        self.param['vehicles'].pop(car_id, None)
        self.vehicles_info.pop(car_id, None)
        self.recent_deleted_car_id = car_id

    def update_param_for_insert(self, extent, location, rz_degree, use_old_id=False, ass_id=-1):
        """
        Add a new vehicle to the current frame and update parameters.

        :param extent: Half-extents of the new vehicle (numpy array: [half-length, half-width, half-height]).
        :param location: Location of the new vehicle (numpy array: [x, y, z]).
        :param rz_degree: Yaw angle of the new vehicle (degrees, around Z-axis).
        :param use_old_id: Boolean indicating if the most recently deleted ID should be reused (if available).
        :param ass_id: Associated ID for the new vehicle (default: -1).
        :return: ID of the newly added vehicle.
        """
        if len(list(self.param['vehicles'].keys())) == 0:
            car_id = 1
        elif use_old_id and self.recent_deleted_car_id != -1:
            car_id = self.recent_deleted_car_id
        else:
            car_id = list(self.param['vehicles'].keys())[-1] + 1

        car_dict = {
            'angle': [0, rz_degree, 0],
            'ass_id': ass_id,
            'center': [0, 0, 0],
            'extent': extent.tolist(),
            'location': location.tolist(),
            'obj_type': 'Car'
        }

        self.param['vehicles'][car_id] = car_dict
        self.load_vehicles_info()
        return car_id


    def save_data_and_label(self, folder_name, save_to_pcd=False):
        """
        Save the current frame's point cloud and parameters to files.

        Directory structure:
        - folder_name
          |- 0 (ego vehicle)
              |- xxx.bin/.pcd (point cloud)
              |- xxx.yaml (parameters)
          |- 1 (cooperative vehicle)
              |- xxx.bin/.pcd (point cloud)
              |- xxx.yaml (parameters)

        :param folder_name: Name of the root folder to save data.
        :param save_to_pcd: Boolean indicating if point cloud should be saved as .pcd (True) or .bin (False).
        """
        saved_dir = os.path.join(self.dataset_config.v2x_dataset_saved_dir, folder_name)

        if self.is_ego:
            saved_path = os.path.join(saved_dir, "0")
        else:
            saved_path = os.path.join(saved_dir, "1")

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        label_saved_path = os.path.join(saved_path, f"{self.bg_index:06d}.yaml")

        if save_to_pcd:
            data_saved_path = os.path.join(saved_path, f"{self.bg_index:06d}.pcd")
            saved_pcd = pc_numpy_2_o3d(self.pc)
            if not saved_pcd.has_colors():
                print("no color data!")
                sys.exit()
            o3d.io.write_point_cloud(data_saved_path, saved_pcd)
        else:
            data_saved_path = os.path.join(saved_path, f"{self.bg_index:06d}.bin")
            saved_pc = self.completed_pc(self.pc[:, :3]).astype(np.float32)
            saved_pc.tofile(data_saved_path)

        with open(label_saved_path, 'w') as outfile:
            yaml.dump(self.param, outfile, default_flow_style=False)

    @staticmethod
    def completed_pc(mixed_pc_three):
        """
        Add a fourth channel (zero intensity) to a 3-channel (XYZ) point cloud.

        :param mixed_pc_three: Numpy array of shape (N, 3) (XYZ coordinates).
        :return: Numpy array of shape (N, 4) (XYZ + zero intensity).
        """
        assert mixed_pc_three.shape[1] == 3

        hang = mixed_pc_three.shape[0]
        b = np.zeros((hang, 1))
        mixed_pc = np.concatenate([mixed_pc_three, b], axis=1)
        return mixed_pc

    @staticmethod
    def pcd_file_to_np(pcd_file):
        """
        Read a .pcd file and convert it to a numpy array (XYZ + intensity).

        :param pcd_file: Path to the .pcd file.
        :return: Tuple containing:
            - pcd: Open3D PointCloud object (for visualization).
            - pcd_np: Numpy array of shape (N, 4) (XYZ + intensity, float32).
        """
        pcd = o3d.io.read_point_cloud(pcd_file)

        xyz = np.asarray(pcd.points)
        # we save the intensity in the first channel
        intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)

        pcd_np = np.hstack((xyz, intensity))

        return np.asarray(pcd_np, dtype=np.float32)