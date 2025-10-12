import copy
import json
import os.path
import random
import argparse
import shutil
import obj_transformation as trans
from logger import CLogger
from config.config import Config
from utils.v2x_object import V2XInfo


def rq1_vis_parser():
    parser = argparse.ArgumentParser(description="rq1 command")
    parser.add_argument('-m', '--method', help="the transformation times of frames", default=1)
    args = parser.parse_args()
    return args


def read_from_json(file_path):
    """
    Read json from file path.

    :param file_path: file path
    :return: the content in JSON format if is read successfully
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Cant read file: {e}")
        print(f)
        return []


def save_to_json(file_path, data_list):
    """
    Save to json.

    :param file_path: file path
    :param file_path: data_list path
    :return:
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data_list, f, indent=4)
    except Exception as e:
        print(f"Cant write to file: {e}")


def update_to_list(data_list, turn_dict):
    is_exist = False
    for i, entry in enumerate(data_list):
        if entry.get("bg_index") == turn_dict.get("bg_index"):
            data_list[i] = turn_dict
            is_exist = True
            break

    if not is_exist:
        data_list.append(turn_dict)


def get_saved_dict(ego_info, cp_info, ego_info_baseline, cp_info_baseline,
                   v2x_ego_id, v2x_cp_id, baseline_ego_id, baseline_cp_id, bg_index):
    if baseline_ego_id == -1:
        saved_dict = {
            "bg_index": bg_index,
            "v2x_ego_center": [0, 0],
            "v2x_cp_center": [0, 0],
            "baseline_ego_center": [0, 0],
            "baseline_cp_center": [0, 0]
        }

        return saved_dict
    baseline_ego_center = list(ego_info_baseline.vehicles_info[baseline_ego_id]["center"])[:2]
    if baseline_cp_id == -1:
        baseline_cp_center = baseline_ego_center
    else:
        baseline_cp_center = list(cp_info_baseline.vehicles_info[baseline_cp_id]["center"])[:2]
    if v2x_ego_id == -1:
        v2x_cp_center = list(cp_info.vehicles_info[v2x_cp_id]["center"][:2])
        v2x_ego_center = v2x_cp_center
    elif v2x_cp_id == -1:
        v2x_ego_center = list(ego_info.vehicles_info[v2x_ego_id]["center"][:2])
        v2x_cp_center = v2x_ego_center
    else:
        v2x_ego_center = list(ego_info.vehicles_info[v2x_ego_id]["center"][:2])
        v2x_cp_center = list(cp_info.vehicles_info[v2x_cp_id]["center"][:2])

    saved_dict = {
        "bg_index": bg_index,
        "v2x_ego_center": v2x_ego_center,
        "v2x_cp_center": v2x_cp_center,
        "baseline_ego_center": baseline_ego_center,
        "baseline_cp_center": baseline_cp_center
    }

    return saved_dict


def rq1_frd(method):
    """
    From the 1st to 9th data scenarios of the test dataset,
    save the original data respectively, and generate new data
    according to the baseline and V2XGen methods.
    The generated results will be used to calculate the FRD index
    through the lidargen project, and this index can measure the fidelity of the generated data.

    :param method: the method to generate data
    :return:
    """
    filename = f"rq1/rq1_cut_center.json"
    cut_data_list = read_from_json(filename)

    # traversal 9 scenes
    for i in range(1, 10):
        OP_TIMES = int(method)

        # load dataset config
        dataset_config = Config(dataset="v2x_dataset", scene=i)
        dataset_config.v2x_dataset_saved_dir = \
            f"{dataset_config.dataset_root}/rq1/random_trans_M{OP_TIMES}"
        scene_data_num = dataset_config.scene_data_num
        index_list = list(range(dataset_config.begin_index,
                                dataset_config.begin_index + scene_data_num))
    # 50 frames from each scene
        random_index_list = random.sample(index_list, 50)

        for bg_index in random_index_list:
            ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
            cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)
            ego_info_baseline = copy.deepcopy(ego_info)
            cp_info_baseline = copy.deepcopy(cp_info)
            ego_info_baseline.pc = ego_info_baseline.pc[:, :3]
            cp_info_baseline.pc = cp_info_baseline.pc[:, :3]

            op_count = 0
            total_car_num = ego_info.get_vehicles_nums()

            selected_car_id = []
            saved_center_dict = {}

            # transformations times of each point cloud frame
            while op_count < OP_TIMES:
                # random select a transformation
                transformation_list = [
                    "insert",
                    "delete",
                    "translation",
                    "scaling",
                    "rotation"
                ]

                transformation = random.choice(transformation_list)

                CLogger.info(f"Background {bg_index}, {transformation} operation, M{op_count + 1}")

                if transformation != "insert":
                    if ego_info.get_vehicles_nums() == 0 or \
                            ego_info.get_vehicles_nums() == len(selected_car_id):
                        op_count += 1
                        continue
                    car_id = random.choice(list(ego_info.vehicles_info.keys()))
                    if car_id in selected_car_id:
                        continue
                    selected_car_id.append(car_id)
                    if len(selected_car_id) == total_car_num:
                        # There is only a cooperative vehicle in the scenario.
                        saved_center_dict = get_saved_dict(ego_info, cp_info, ego_info_baseline, cp_info_baseline,
                                                           -1, -1, -1, -1, bg_index)
                        op_count += 1
                        continue

                success_flag = False

                # v2x and baseline transformation
                if transformation == "insert":
                    success_flag, v2x_ego_id, v2x_cp_id, base_ego_id, base_cp_id = \
                        trans.vehicle_insert(ego_info, cp_info, ego_info_baseline, cp_info_baseline)
                    # record center for cut
                    if success_flag:
                        saved_center_dict = get_saved_dict(ego_info, cp_info, ego_info_baseline, cp_info_baseline,
                                                           v2x_ego_id, v2x_cp_id, base_ego_id, base_cp_id,
                                                           bg_index)
                elif transformation == "delete":
                    success_flag, v2x_ego_center, v2x_cp_center, baseline_ego_center, baseline_cp_center = \
                        trans.vehicle_delete(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id)
                    # record center for cut
                    if success_flag:
                        saved_center_dict = {
                            "bg_index": bg_index,
                            "v2x_ego_center": v2x_ego_center,
                            "v2x_cp_center": v2x_cp_center,
                            "baseline_ego_center": baseline_ego_center,
                            "baseline_cp_center": baseline_cp_center
                        }
                elif transformation == "translation":
                    success_flag, v2x_ego_id, v2x_cp_id, base_ego_id, base_cp_id = \
                        trans.vehicle_translation(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id)
                    # record center for cut
                    if success_flag:
                        saved_center_dict = get_saved_dict(ego_info, cp_info, ego_info_baseline, cp_info_baseline,
                                                           v2x_ego_id, v2x_cp_id, base_ego_id, base_cp_id,
                                                           bg_index)
                elif transformation == "scaling":
                    success_flag, v2x_ego_id, v2x_cp_id, base_ego_id, base_cp_id = \
                            trans.vehicle_scaling(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id)
                    # record center for cut
                    if success_flag:
                        saved_center_dict = get_saved_dict(ego_info, cp_info, ego_info_baseline, cp_info_baseline,
                                                           v2x_ego_id, v2x_cp_id, base_ego_id, base_cp_id,
                                                           bg_index)
                else:
                    success_flag, v2x_ego_id, v2x_cp_id, base_ego_id, base_cp_id = \
                            trans.vehicle_rotation(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id)
                    # record center for cut
                    if success_flag:
                        saved_center_dict = get_saved_dict(ego_info, cp_info, ego_info_baseline, cp_info_baseline,
                                                           v2x_ego_id, v2x_cp_id, base_ego_id, base_cp_id, bg_index)
                if not success_flag:
                    continue

                op_count += 1

            ego_info.save_data_and_label(f"v2x_scene{i}")
            cp_info.save_data_and_label(f"v2x_scene{i}")
            ego_info_baseline.save_data_and_label(f"baseline_scene{i}")
            cp_info_baseline.save_data_and_label(f"baseline_scene{i}")

            # if cut data, record center of every frame
            """
            1. record bg_index
            2.  insert: ego_center, cp_center [v2x + baseline]
                delete: ego_center, cp_center
                rotation: ego_center, cp_center
                translation: ego_center, cp_center
                scaling: ego_center, cp_center
            """
            update_to_list(cut_data_list, saved_center_dict)
            save_to_json(filename, cut_data_list)

            # copy the original data
            ego_ori_path = f"{dataset_config.dataset_root}/v2v_test/0/velodyne/{bg_index:06d}.bin"
            cp_ori_path = f"{dataset_config.dataset_root}/v2v_test/1/velodyne/{bg_index:06d}.bin"
            ego_save_folder = f"{dataset_config.dataset_root}/rq1/ori_M{OP_TIMES}/selected_ori_scene{i}/0"
            cp_save_folder = f"{dataset_config.dataset_root}/rq1/ori_M{OP_TIMES}/selected_ori_scene{i}/1"
            if not os.path.exists(ego_save_folder):
                os.makedirs(ego_save_folder)
            if not os.path.exists(cp_save_folder):
                os.makedirs(cp_save_folder)

            shutil.copy(ego_ori_path, ego_save_folder)
            shutil.copy(cp_ori_path, cp_save_folder)


if __name__ == '__main__':
    cmd_args = rq1_vis_parser()
    # generate data for frd
    rq1_frd(cmd_args.method)
