import os
import argparse
import random
import shutil
import json


def rq2_split_parser():
    parser = argparse.ArgumentParser(description="rq2 command")
    parser.add_argument('-d', '--dataset', help="the dataset path")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    1. Half of the sequences are randomly selected and saved as a training set for retrain
    and a test set for testing.
    2. copy selected data to new folder velodyne/labels/predictions.
    """
    # 创建 rq2/selected_number.json 记录划分结果
    # 一半作为 selected (用于后续变换生成训练/重训集)，一半作为 test (测试集)
    # 将原始点云（.pcd/.bin）、标签（.yaml）、预测结果等文件，按索引分别拷贝到 
    # pre_trans_dataset 和 test_dataset 文件夹中
    cmd_args = rq2_split_parser()
    all_index_list = list(range(1, 1994))

    selected_index_list = random.sample(all_index_list, len(all_index_list) // 2)   # train dataset
    test_index_list = list(set(all_index_list) - set(selected_index_list))          # test dataset

    # save sequences
    with open("rq2/selected_number.json", "w") as file:
        json.dump({
            "selected": selected_index_list,
            "test": test_index_list
        }, file, indent=4)

    dataset_root = cmd_args.dataset
    v2v_test_root = os.path.join(dataset_root, "v2x_dataset")
    rq2_dataset_root = os.path.join(dataset_root, "rq2")

    trans_folder = f"{rq2_dataset_root}/pre_trans_dataset"
    test_folder = f"{rq2_dataset_root}/test_dataset"

    for sub_folder in ["0", "1"]:
        for folder_type in ["velodyne", "yaml", "predictions", "pcd"]:
            files = os.listdir(os.path.join(v2v_test_root, sub_folder, folder_type))

            for file in files:
                file_number = int(file.split('.')[0])
                if file_number in selected_index_list:
                    target_folder = trans_folder
                else:
                    target_folder = test_folder

                source_path = os.path.join(v2v_test_root, sub_folder, folder_type, file)
                des_folder = os.path.join(target_folder, sub_folder, folder_type)
                des_path = os.path.join(target_folder, sub_folder, folder_type, file)

                if not os.path.exists(des_folder):
                    os.makedirs(des_folder)

                shutil.copy(source_path, des_path)

