import os
import csv
import time
import yaml
import shutil
import argparse
import numpy as np
from glob import glob


def args_parser():
    parser = argparse.ArgumentParser(description="rq1 command")
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                    help='输出的 v2x 数据集目录（默认: dataset_dir 上级目录下的 v2x_dataset）')
    parser.add_argument('--metrics_file', type=str, default=None,
                    help='逐帧处理指标 CSV（默认: output_dir/frame_metrics.csv）')
    args = parser.parse_args()
    return args


def read_ori_pcd(input_path):
    """
    1. read x, y, z, intensity of every point in pcd files
    2. fit to semanticKitti
    """
    lidar = []
    with open(input_path, 'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)


# def convert2bin(input_pcd_dir, output_bin_dir):
#     file_list = os.listdir(input_pcd_dir)
#     if not os.path.exists(output_bin_dir):
#         os.makedirs(output_bin_dir)
#     for file in file_list:
#         (filename, extension) = os.path.splitext(file)
#         velodyne_file = os.path.join(input_pcd_dir, filename) + '.pcd'
#         p_xyzi = read_ori_pcd(velodyne_file)
#         p_xyzi = p_xyzi.reshape((-1, 4)).astype(np.float32)
#         min_val = np.amin(p_xyzi[:, 3])
#         max_val = np.amax(p_xyzi[:, 3])
#         p_xyzi[:, 3] = (p_xyzi[:, 3] - min_val)/(max_val-min_val)
#         p_xyzi[:, 3] = np.round(p_xyzi[:, 3], decimals=2)
#         p_xyzi[:, 3] = np.minimum(p_xyzi[:, 3], 0.99)
#         velodyne_file_new = os.path.join(output_bin_dir, filename) + '.bin'
#         p_xyzi.tofile(velodyne_file_new)


def convert2bin(input_pcd_dir, output_bin_dir, metrics_writer=None, stream=""):
    # pcd files to bin
    # file_list = os.listdir(input_pcd_dir)
    # file_list = glob(input_pcd_dir)
    file_list = sorted(glob(input_pcd_dir))
    if not os.path.exists(output_bin_dir):
        os.makedirs(output_bin_dir)

    # 防止上一次运行产生的多余 BIN 残留
    clear_files(output_bin_dir, "*.bin")

    total_elapsed_s = 0.0
    total_points = 0
    for i, file in enumerate(file_list):
        frame_start = time.perf_counter()
        # (filename, extension) = os.path.splitext(file)
        # velodyne_file = os.path.join(input_pcd_dir, filename) + '.pcd'
        p_xyzi = read_ori_pcd(file)
        p_xyzi = p_xyzi.reshape((-1, 4)).astype(np.float32)
        min_val = np.amin(p_xyzi[:, 3])
        max_val = np.amax(p_xyzi[:, 3])
        p_xyzi[:, 3] = (p_xyzi[:, 3] - min_val)/(max_val-min_val)
        p_xyzi[:, 3] = np.round(p_xyzi[:, 3], decimals=2)
        p_xyzi[:, 3] = np.minimum(p_xyzi[:, 3], 0.99)
        velodyne_file_new = os.path.join(output_bin_dir, f"{i + 1:06d}") + '.bin'
        p_xyzi.tofile(velodyne_file_new)

        elapsed_s = time.perf_counter() - frame_start
        fps = 1.0 / elapsed_s if elapsed_s > 0 else float("inf")
        total_elapsed_s += elapsed_s
        total_points += len(p_xyzi)
        if metrics_writer is not None:
            metrics_writer.writerow({
                "stream": stream,
                "frame": i + 1,
                "input_file": file,
                "output_file": velodyne_file_new,
                "point_count": len(p_xyzi),
                "elapsed_ms": f"{elapsed_s * 1000:.3f}",
                "fps": f"{fps:.3f}",
            })
        print(
            f"[{stream}] frame {i + 1}/{len(file_list)}: "
            f"{elapsed_s * 1000:.3f} ms, {fps:.3f} FPS"
        )

    total_frames = len(file_list)
    avg_ms = (total_elapsed_s / total_frames * 1000
              if total_frames else 0.0)
    overall_fps = (total_frames / total_elapsed_s
                   if total_elapsed_s > 0 else 0.0)
    print(
        f"[{stream}] summary: {total_frames} frames, "
        f"{total_elapsed_s:.3f} sec, {avg_ms:.3f} ms/frame, "
        f"{overall_fps:.3f} FPS"
    )
    return {
        "stream": stream,
        "total_frames": total_frames,
        "total_points": total_points,
        "total_elapsed_s": total_elapsed_s,
        "avg_ms": avg_ms,
        "overall_fps": overall_fps,
    }


def clear_files(directory, file_pattern):
    """删除指定目录中符合规则的旧文件。"""
    os.makedirs(directory, exist_ok=True)
    old_files = glob(os.path.join(directory, file_pattern))

    for old_file in old_files:
        os.remove(old_file)

    print(
        f"清理目录: {directory}, "
        f"删除 {len(old_files)} 个 {file_pattern} 文件"
    )
def copy_files(input_dir, output_dir, file_format="pcd"):
    file_list = sorted(glob(input_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, source_file in enumerate(file_list):
        # print(source_file)
        des_file = os.path.join(output_dir, f"{i + 1:06d}.") + file_format
        shutil.copy(source_file, des_file)


if __name__ == '__main__':
    opt = args_parser()
    test_dir = opt.dataset_dir
    dataset_root = os.path.dirname(test_dir)

    v2x_dataset_dir = opt.output_dir or os.path.join(dataset_root, "v2x_dataset")

    if not os.path.exists(v2x_dataset_dir):
        os.makedirs(v2x_dataset_dir)

    metrics_file = opt.metrics_file or os.path.join(
        v2x_dataset_dir, "frame_metrics.csv")
    metrics_parent = os.path.dirname(os.path.abspath(metrics_file))
    os.makedirs(metrics_parent, exist_ok=True)
    metrics_handle = open(metrics_file, "w", newline="")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=[
        "stream", "frame", "input_file", "output_file", "point_count",
        "elapsed_ms", "fps",
    ])
    metrics_writer.writeheader()
    summaries = []

    try:
        for file_format in ["pcd", "yaml"]:
            ego_files = os.path.join(test_dir, f"*/0/*.{file_format}")
            cp_files = os.path.join(test_dir, f"*/1/*.{file_format}")

            des_ego_dir = os.path.join(v2x_dataset_dir, f"0/{file_format}")
            des_cp_dir = os.path.join(v2x_dataset_dir, f"1/{file_format}")

            # copy pcds and labels
            copy_files(ego_files, des_ego_dir, file_format)
            copy_files(cp_files, des_cp_dir, file_format)

            # pcd to bin (velodyne)
            if file_format == "pcd":
                ego_bin_dir = os.path.join(v2x_dataset_dir, f"0/velodyne")
                cp_bin_dir = os.path.join(v2x_dataset_dir, f"1/velodyne")
                summaries.append(convert2bin(
                    ego_files, ego_bin_dir, metrics_writer, stream="ego"))
                metrics_handle.flush()
                summaries.append(convert2bin(
                    cp_files, cp_bin_dir, metrics_writer, stream="cp"))
                metrics_handle.flush()
    finally:
        metrics_handle.close()

    print(f"逐帧处理指标已保存到: {metrics_file}")

    ego_summary = summaries[0]
    cp_summary = summaries[1]

    ego_frames = ego_summary["total_frames"]
    cp_frames = cp_summary["total_frames"]

    # 一个协同场景必须同时包含一帧 ego 和一帧 cp
    if ego_frames != cp_frames:
        raise ValueError(
            f"ego 和 cp 帧数不一致，无法组成完整的协同场景："
            f"ego={ego_frames}, cp={cp_frames}"
        )

    # 协同场景数量不是 ego + cp，而是成对后的数量
    combined_frames = ego_frames
    # 当前程序顺序处理 ego 和 cp，因此场景总耗时为两者之和
    combined_elapsed_s = (ego_summary["total_elapsed_s"] + cp_summary["total_elapsed_s"])
    combined_points = (ego_summary["total_points"] + cp_summary["total_points"])

    summaries.append({
        "stream": "all",
        "total_frames": combined_frames,
        "total_points": combined_points,
        "total_elapsed_s": combined_elapsed_s,
        "avg_ms": (combined_elapsed_s / combined_frames * 1000
                   if combined_frames else 0.0),
        "overall_fps": (combined_frames / combined_elapsed_s
                        if combined_elapsed_s > 0 else 0.0),
    })
    summary_file = os.path.join(metrics_parent, "processing_summary.csv")
    with open(summary_file, "w", newline="") as summary_handle:
        summary_writer = csv.DictWriter(summary_handle, fieldnames=[
            "stream", "total_frames", "total_points", "total_elapsed_s",
            "avg_ms", "overall_fps",
        ])
        summary_writer.writeheader()
        for item in summaries:
            summary_writer.writerow({
                "stream": item["stream"],
                "total_frames": item["total_frames"],
                "total_points": item["total_points"],
                "total_elapsed_s": f'{item["total_elapsed_s"]:.6f}',
                "avg_ms": f'{item["avg_ms"]:.3f}',
                "overall_fps": f'{item["overall_fps"]:.3f}',
            })
    all_summary = summaries[-1]
    print(
        f"[all] summary: {all_summary['total_frames']} frames, "
        f"{all_summary['total_elapsed_s']:.3f} sec, "
        f"{all_summary['avg_ms']:.3f} ms/frame, "
        f"{all_summary['overall_fps']:.3f} FPS"
    )
    print(f"总体处理指标已保存到: {summary_file}")

    # get semantic predictions
    # 1. copy bin files conforms to the semanticKitti format
    ego_bin_dir = os.path.join(v2x_dataset_dir, f"0/velodyne/*.bin")
    cp_bin_dir = os.path.join(v2x_dataset_dir, f"1/velodyne/*.bin")

    semantic_dir = os.path.join(dataset_root, "semantic")

    des_bin_ego_dir = os.path.join(semantic_dir, "semanticKitti/sequences/11/velodyne")
    des_bin_cp_dir = os.path.join(semantic_dir, "semanticKitti/sequences/12/velodyne")

    clear_files(des_bin_ego_dir, "*.bin")
    clear_files(des_bin_cp_dir, "*.bin")

    copy_files(ego_bin_dir, des_bin_ego_dir, "bin")
    copy_files(cp_bin_dir, des_bin_cp_dir, "bin")

    # 2. semantic segmentation
    cmd1 = "cd ./third/SalsaNext/train/tasks/semantic"
    cmd2 = f"python infer.py -d '{semantic_dir}/semanticKitti' -m pretrained -l '{semantic_dir}/result' -s validation"

    # 清理上一次语义推理的预测结果
    semantic_prediction_ego_dir = os.path.join(
        semantic_dir, "result/sequences/11/predictions")
    semantic_prediction_cp_dir = os.path.join(
        semantic_dir, "result/sequences/12/predictions")

    clear_files(semantic_prediction_ego_dir, "*.label")
    clear_files(semantic_prediction_cp_dir, "*.label")

    os.system(f"{cmd1} && {cmd2}")

    # 3. copy the results of the semantic segmentation predictions
    ego_prediction_results = os.path.join(semantic_dir, f'{semantic_dir}/result/sequences/11/predictions/*.label')
    cp_prediction_results = os.path.join(semantic_dir, f'{semantic_dir}/result/sequences/12/predictions/*.label')

    des_prediction_ego_dir = os.path.join(v2x_dataset_dir, f"0/predictions")
    des_prediction_cp_dir = os.path.join(v2x_dataset_dir, f"1/predictions")

    clear_files(des_prediction_ego_dir, "*.label")
    clear_files(des_prediction_cp_dir, "*.label")

    copy_files(ego_prediction_results, des_prediction_ego_dir, "label")
    copy_files(cp_prediction_results, des_prediction_cp_dir, "label")

    # 4. save the dataset path in the config
    dataset_config = {"dataset_path": dataset_root}

    with open("./config/dataset_config.yml", "w") as config_file:
        yaml.dump(dataset_config, config_file, default_flow_style=False)
