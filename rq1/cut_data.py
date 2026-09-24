import os
import json
import argparse
import numpy as np
import utils.v2X_file as v2v
from logger import CLogger
from config.config import Config
from utils.v2X_file import read_Bin_PC


def data_cut_parser():
    parser = argparse.ArgumentParser(description="rq1 data cut command")
    parser.add_argument('-t', '--transform',
                        help="select a transform operation, insert/delete/translation/scaling/rotation")
    parser.add_argument('-s', '--size', help="cut size of data (sxs)", default=10)
    args = parser.parse_args()
    return args


def read_from_json(file_path):
    """ read json """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Cant read file: {e}")
        print(f)
        return []


def load_pc(bg_index, pc_path):
    bg_pc_path = os.path.join(pc_path, f"{bg_index:06d}.bin")
    bg_xyz = read_Bin_PC(bg_pc_path)
    return bg_xyz


def save_data(data, save_path): # 补全点云维度（通常是 4 维）并保存为 .bin
    save_path_folder = os.path.dirname(save_path)
    if not os.path.exists(save_path_folder):
        os.makedirs(save_path_folder)
    complete_data = v2v.complet_pc(data).astype(np.float32)
    complete_data.tofile(save_path)


def crop_point_cloud(pc, center, size): #在点云中截取以 center 为中心、size * size范围的方形区域
    x, y = center

    x_min = x - size / 2
    x_max = x + size / 2
    y_min = y - size / 2
    y_max = y + size / 2

    cropped_pc = pc[
        (pc[:, 0] >= x_min) & (pc[:, 0] <= x_max) &
        (pc[:, 1] >= y_min) & (pc[:, 1] <= y_max)
        ]

    # print(cropped_pc.shape)
    return cropped_pc


if __name__ == '__main__':
    cmd_args = data_cut_parser()

    is_unconditional = True

    # transformation and cut size
    cut_size = int(cmd_args.size)
    cut_transformation = cmd_args.transform

    dataset_config = Config(dataset="rq1")
    select_data_num = dataset_config.select_data_num
    index_list = list(range(1, select_data_num + 1))

    cut_label_path = f"rq1/cut_center_{cut_transformation}.json"
    cut_data_list = read_from_json(cut_label_path)

    if is_unconditional:
        v2x_dir = f"{dataset_config.dataset_path}"
        baseline_dir = f"{dataset_config.dataset_path}"
        v2x_ego_dir = f"{v2x_dir}/0/velodyne"
        v2x_cp_dir = f"{v2x_dir}/1/velodyne"
        baseline_ego_dir = f"{baseline_dir}/0/velodyne"
        baseline_cp_dir = f"{baseline_dir}/1/velodyne"
    else:
        dataset_dir = f"{dataset_config.dataset_path}/times1"
        baseline_dir = f"{dataset_dir}/baseline_{cut_transformation}_times1"
        v2x_dir = f"{dataset_dir}/v2x_{cut_transformation}_times1"
        v2x_ego_dir = f"{v2x_dir}/0"
        v2x_cp_dir = f"{v2x_dir}/1"
        baseline_ego_dir = f"{baseline_dir}/0"
        baseline_cp_dir = f"{baseline_dir}/1"

    for bg_index in range(1, select_data_num + 1):
        CLogger.info(f"Background {bg_index}")

        for i, entry in enumerate(cut_data_list):
            if entry.get("bg_index") == bg_index:
                v2x_ego_center = entry["v2x_ego_center"]
                v2x_cp_center = entry["v2x_cp_center"]
                baseline_ego_center = entry["baseline_ego_center"]
                baseline_cp_center = entry["baseline_cp_center"]

                v2x_ego_pc = v2v.load_pc(bg_index, v2x_ego_dir)
                v2x_cp_pc = v2v.load_pc(bg_index, v2x_cp_dir)
                baseline_ego_pc = v2v.load_pc(bg_index, baseline_ego_dir)
                baseline_cp_pc = v2v.load_pc(bg_index, baseline_cp_dir)

                # cut data
                cut_v2x_ego_pc = crop_point_cloud(v2x_ego_pc, v2x_ego_center, cut_size)
                cut_v2x_cp_pc = crop_point_cloud(v2x_cp_pc, v2x_cp_center, cut_size)
                cut_baseline_ego_pc = crop_point_cloud(baseline_ego_pc, baseline_ego_center, cut_size)
                cut_baseline_cp_pc = crop_point_cloud(baseline_cp_pc, baseline_cp_center, cut_size)

                # save data
                if is_unconditional:
                    dataset_saved_dir = f"{dataset_config.dataset_path}/ori_cut_{cut_size}"
                else:
                    dataset_saved_dir = f"{dataset_config.dataset_path}/cut_{cut_size}"
                baseline_ego_saved_dir = f"{dataset_saved_dir}/baseline_{cut_transformation}_times1/0/{bg_index:06d}.bin"
                baseline_cp_saved_dir = f"{dataset_saved_dir}/baseline_{cut_transformation}_times1/1/{bg_index:06d}.bin"
                v2x_ego_saved_dir = f"{dataset_saved_dir}/v2x_{cut_transformation}_times1/0/{bg_index:06d}.bin"
                v2x_cp_saved_dir = f"{dataset_saved_dir}/v2x_{cut_transformation}_times1/1/{bg_index:06d}.bin"

                save_data(cut_v2x_ego_pc, v2x_ego_saved_dir)
                save_data(cut_v2x_cp_pc, v2x_cp_saved_dir)
                save_data(cut_baseline_ego_pc, baseline_ego_saved_dir)
                save_data(cut_baseline_cp_pc, baseline_cp_saved_dir)







