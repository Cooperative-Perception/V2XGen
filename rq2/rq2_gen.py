import json
import random
import argparse
import obj_transformation as trans
from logger import CLogger
from config.config import Config
from utils.v2x_object import V2XInfo


def rq2_vis_parser():
    parser = argparse.ArgumentParser(description="rq1 command")
    parser.add_argument('-m', '--method', help="the transformation times of frames", default=1)
    parser.add_argument('-g', '--generate', help="is generate data", default=True)
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


def rq2_gen(method, is_gen=True):
    """
    RQ2: Perform the method transformation on each selected data and
    save the transformation result.

    :param method: transformation times
    :param is_gen: original data or generated data
    :return:
    """
    OP_TIMES = int(method)

    # load dataset config
    dataset_config = Config(dataset="rq2")

    if not is_gen:
        dataset_config.v2x_dataset_saved_dir = \
            f"{dataset_config.dataset_root}/rq2/rq2_gen/ori_data"

        # dataset_config.v2x_dataset_saved_dir = \
        #     f"{dataset_config.dataset_root}/rq3/rq3_test/ori_data"
        # dataset_config.dataset_path = os.path.join(dataset_config.dataset_root, "rq3/test_dataset")
    else:
        dataset_config.v2x_dataset_saved_dir = \
            f"{dataset_config.dataset_root}/rq_eval/rq2_gen/trans_M{OP_TIMES}"

        # get the occlusion and distance label of rq3 test data
        # dataset_config.v2x_dataset_saved_dir = \
        #         f"{dataset_config.dataset_root}/rq3/rq3_test/test_M{OP_TIMES}"
        # dataset_config.dataset_path = os.path.join(dataset_config.dataset_root, "rq3/test_dataset")

    # load selected data index list
    selected_index_list = read_from_json("rq_eval/selected_number.json")["test"]
    trans_index_list = sorted(selected_index_list)

    for bg_index in trans_index_list:
        ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
        cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)

        op_count = 0
        total_car_num = ego_info.get_vehicles_nums()

        selected_car_id = []

        # set the times of operation per frame data
        while op_count < OP_TIMES:
            # label the original data
            if not is_gen:
                break

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

            car_id = 0

            if transformation != "insert":
                # pick a random car from the ego scene
                if ego_info.get_vehicles_nums() == 0 or \
                        ego_info.get_vehicles_nums() == len(selected_car_id):
                    op_count += 1
                    continue
                car_id = random.choice(list(ego_info.vehicles_info.keys()))
                if car_id in selected_car_id:
                    continue
                selected_car_id.append(car_id)
                if len(selected_car_id) == total_car_num:
                    op_count += 1
                    continue

            success_flag = False

            # v2x and baseline transformation
            if transformation == "insert":
                success_flag = trans.vehicle_insert(ego_info, cp_info)
            elif transformation == "delete":
                success_flag = trans.vehicle_delete(ego_info, cp_info, car_id)
            elif transformation == "translation":
                success_flag = trans.vehicle_translation(ego_info, cp_info, car_id)
            elif transformation == "scaling":
                success_flag = trans.vehicle_scaling(ego_info, cp_info, car_id)
            else:
                success_flag = trans.vehicle_rotation(ego_info, cp_info, car_id)
            if not success_flag:
                continue

            op_count += 1

        # scan each vehicle in the label vehicle list, calculate the occlusion rate and distance, and save it in the tag
        trans.label_complete_for_ego(ego_info, cp_info)
        trans.label_complete_for_cp(ego_info, cp_info)

        ego_info.save_data_and_label("", save_to_pcd=True)
        cp_info.save_data_and_label("", save_to_pcd=True)


if __name__ == '__main__':
    cmd_args = rq2_vis_parser()
    # generate data for frd
    rq2_gen(cmd_args.method, True)
