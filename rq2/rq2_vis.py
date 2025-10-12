import json
import os.path
import random
import argparse
import obj_transformation as trans
from config.config import Config
from utils.v2x_object import V2XInfo


def rq2_vis_parser():
    parser = argparse.ArgumentParser(description="rq1 command")
    parser.add_argument('-s', '--scene', help="the scene of dataset", default=1)
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
        return {}
    except Exception as e:
        print(f"Cant read file: {e}")
        print(f)
        return {}


def rq2_vis(scene=1):
    """
    Visualization of RQ2: Visualize the generated data,
    which needs to cover road completion and occluded area insertion operations.
    """
    # load dataset config
    dataset_config = Config(dataset="rq_eval")
    dataset_config.dataset_path = os.path.join(dataset_config.dataset_root, "rq3/test_dataset")

    # begin_index = dataset_config.begin_index
    selected_index_list = read_from_json("rq_eval/selected_number.json")["selected"]
    trans_index_list = sorted(selected_index_list)

    for bg_index in trans_index_list:
        ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
        cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)

        car_id = random.choice(list(ego_info.vehicles_info.keys()))
        print(list(ego_info.vehicles_info.keys()))
        car_id = 5
        success_flag = trans.vehicle_delete(ego_info, cp_info, car_id)


if __name__ == '__main__':
    cmd_args = rq2_vis_parser()
    # generate data for frd
    rq2_vis(int(cmd_args.scene))
