import os
import shutil
import json
from datetime import datetime

import numpy as np
import torch
import random

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
                    left_range=-float('inf'), right_range=float('inf'),
                    gt_object_ids=None):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    right_range : float
        The evaluarion range right bound
    left_range : float
        The evaluation range left bound
    gt_object_ids : list
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []

    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        det_polygon_list_origin = list(common_utils.convert_format(det_boxes))
        gt_polygon_list_origin = list(common_utils.convert_format(gt_boxes))
        det_polygon_list = []
        gt_polygon_list = []
        det_score_new = []
        # remove the bbx out of range
        for i in range(len(det_polygon_list_origin)):
            det_polygon = det_polygon_list_origin[i]
            distance = np.sqrt(det_polygon.centroid.x ** 2 +
                               det_polygon.centroid.y ** 2)
            if left_range < distance < right_range:
                det_polygon_list.append(det_polygon)
                det_score_new.append(det_score[i])

        for i in range(len(gt_polygon_list_origin)):
            gt_polygon = gt_polygon_list_origin[i]
            distance = np.sqrt(gt_polygon.centroid.x ** 2 +
                               gt_polygon.centroid.y ** 2)
            if left_range < distance < right_range:
                gt_polygon_list.append(gt_polygon)

        gt = len(gt_polygon_list)
        det_score_new = np.array(det_score_new)
        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score_new)

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

            # TODO: pop successfully pred gt box
            gt_object_ids.pop(gt_index)
    else:
        gt = gt_boxes.shape[0]
    # result_stat[iou_thresh]['fp'] += fp
    # result_stat[iou_thresh]['tp'] += tp
    # result_stat[iou_thresh]['gt'] += gt
    result_stat[iou_thresh]['tp'].append(tp)
    result_stat[iou_thresh]['fp'].append(fp)
    result_stat[iou_thresh]['gt'].append(gt)

    return fp, tp, gt, gt_object_ids


def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, range=""):
    dump_dict = {}
    file_name = 'eval.yaml' if range == "" else range + '_eval.yaml'
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, file_name))

    # print('The range is %s, '
    #       'The Average Precision at IOU 0.5 is %.3f, '
    #       'The Average Precision at IOU 0.7 is %.3f' % (range, ap_50, ap_70))

    return round(ap_50, 3)


def get_occ_error(ego_gen_param, cp_gen_param, false_pred_ids):
    total_occ = 0
    occ_error = 0
    occ_threshold = 0

    for car_id, param_dict in ego_gen_param.items():
        if param_dict['ego_occlusion_rate'] > occ_threshold:
            if car_id in false_pred_ids:
                occ_error += 1
            total_occ += 1

    for car_id, param_dict in cp_gen_param.items():
        if param_dict['ego_occlusion_rate'] > occ_threshold:
            if car_id in false_pred_ids:
                occ_error += 1
            total_occ += 1

    return occ_error, total_occ


def get_long_distance_error(ego_gen_param, cp_gen_param, false_pred_ids, distance_k=50):
    long_distance_error = 0
    total_long_distance = 0

    for car_id, param_dict in ego_gen_param.items():
        if param_dict['ego_distance'] > distance_k:
            if car_id in false_pred_ids:
                long_distance_error += 1
            total_long_distance += 1

    for car_id, param_dict in cp_gen_param.items():
        if param_dict['ego_distance'] > distance_k:
            if car_id in false_pred_ids:
                long_distance_error += 1
            total_long_distance += 1

    return long_distance_error, total_long_distance


def method_eval_result(method_stat, result_stat, model_dir, scale, is_save=False, dataset_dir=None, save_path=None, model=None,
                       ori_dataset_dir=None, ori_scale=0.15, ori_selected_timestamps=None):
    """
    1. Random select method
    2. CooTest select method
    3. V2x-Gen select method
    :param method_stat:
    :param result_stat:
    :param model_dir:
    :param scale:
    :param is_save:
    :param dataset_dir:
    :param save_path:
    :param model:
    :return:
    """
    total_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0},
                         'occ_error': 0,
                         'dis_error': 0,
                         'total_occ': 0,
                         'total_dis': 0,
                         'timestamp': []}
    cootest_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                           0.7: {'tp': [], 'fp': [], 'gt': 0},
                           'occ_error': 0,
                           'dis_error': 0,
                           'total_occ': 0,
                           'total_dis': 0,
                           'timestamp': []}
    random_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0},
                          'occ_error': 0,
                          'dis_error': 0,
                          'total_occ': 0,
                          'total_dis': 0,
                          'timestamp': []}
    gen_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                       0.7: {'tp': [], 'fp': [], 'gt': 0},
                       'occ_error': 0,
                       'dis_error': 0,
                       'total_occ': 0,
                       'total_dis': 0,
                       'timestamp': []}

    # scene timestamp intervals
    intervals = [(0, 147), (147, 261), (261, 405), (405, 603), (603, 783),
                 (783, 1093), (1093, 1397), (1397, 1618), (1618, 1993)]
    split_scene = False

    num = len(result_stat[0.5]['tp'])   # sum data nums

    # random select
    if not split_scene:
        random_select_indices = sorted(random.sample(range(num), int(num * scale)))
    else:
        random_select_indices = []
        for start, end in intervals:
            scene_indices = []
            timestamp_list = result_stat['timestamp']

            for i, timestamp in enumerate(timestamp_list):
                if start <= int(timestamp[1]) < end:
                    scene_indices.append(i)

            select_count = int(len(scene_indices) * scale)
            random_select_indices += random.sample(scene_indices, select_count)

    # CooTest select
    cootest_stat_list = method_stat['cootest']
    # Normalized data
    max_param = max(cootest_stat_list)
    min_param = min(cootest_stat_list)
    # normalized_params_list = [-(x - min_param) / (max_param - min_param) for x in cootest_stat_list]
    if max_param == min_param:
        normalized_params_list = [0 for x in cootest_stat_list]
    else:
        normalized_params_list = [-(x-min_param)/(max_param-min_param) for x in cootest_stat_list]
    select_number = int(len(normalized_params_list) * scale)

    # V2X-Gen select
    gen_stat_list = method_stat['v2x_gen']

    if split_scene:
        cootest_select_indices = select_scene_scores(result_stat, normalized_params_list, intervals, scale)
        gen_select_indices = select_scene_scores(result_stat, gen_stat_list, intervals, scale)
    else:
        cootest_select_indices = sorted(range(len(normalized_params_list)),
                                        key=lambda i: normalized_params_list[i],
                                        reverse=True)[:select_number]
        gen_select_indices = sorted(range(len(gen_stat_list)),
                                    key=lambda i: gen_stat_list[i],
                                    reverse=True)[:select_number]

    print(len(random_select_indices), len(cootest_select_indices), len(gen_select_indices))

    get_part_list_stat(result_stat, cootest_select_indices, cootest_result_stat)
    get_part_list_stat(result_stat, random_select_indices, random_result_stat)
    get_part_list_stat(result_stat, gen_select_indices, gen_result_stat)
    get_part_list_stat(result_stat, range(num), total_result_stat)

    print("------------------------------------------------------")
    print(f"scale = {scale}")
    print("------------------------------------------------------")
    v2x_select_eval(total_result_stat, model_dir, 'total')
    print("------------------------------------------------------")
    v2x_select_eval(random_result_stat, model_dir, 'random')
    print("------------------------------------------------------")
    v2x_select_eval(cootest_result_stat, model_dir, 'cootest')
    print("------------------------------------------------------")
    v2x_select_eval(gen_result_stat, model_dir, 'gen')
    print("------------------------------------------------------")

    # save result
    # if is_save:
    #     pass
    #     save_selected_data_and_label(cootest_result_stat['timestamp'], dataset_dir, save_path, f'coo_test/{scale}/{model}/select')
    #     save_selected_data_and_label(random_result_stat['timestamp'], dataset_dir, save_path, f'random/{scale}/{model}/select')
    #     save_selected_data_and_label(gen_result_stat['timestamp'], dataset_dir, save_path, f'v2x_gen/{scale}/{model}/select')
    if is_save:
        method_save_configs = [
            ('coo_test', cootest_result_stat['timestamp'], f'coo_test/{scale}/{model}/select'),
            ('random', random_result_stat['timestamp'], f'random/{scale}/{model}/select'),
            ('v2x_gen', gen_result_stat['timestamp'], f'v2x_gen/{scale}/{model}/select')
        ]

        ori_frame_ids_dict = {}

        # 保存变换数据
        for method_name, selected_timestamps, method_save_path in method_save_configs:
            save_selected_data_and_label(selected_timestamps, dataset_dir, save_path, method_save_path)
    
        # 统一保存原始数据（所有方法共享）
        if ori_dataset_dir is not None and ori_selected_timestamps is not None:
            for method_name, _, method_save_path in method_save_configs:
                save_selected_ori_data(
                    ori_dataset_dir,
                    save_path,
                    method_save_path,
                    ori_selected_timestamps
                )
    
        # 统计分布（ori 只统计一次）
        save_method_selection_distribution(
            {
                'coo_test': cootest_result_stat['timestamp'],
                'random': random_result_stat['timestamp'],
                'v2x_gen': gen_result_stat['timestamp']
            },
            save_path,
            scale,
            model,
            intervals,
            ori_selected_timestamps=ori_selected_timestamps  # 传入统一的 ori 数据
        )

def save_method_selection_distribution(method_timestamp_dict, save_dir, scale, model,
                                       scene_intervals, ori_selected_timestamps=None):
    distribution = {}

    for method_name, timestamps in method_timestamp_dict.items():
        scene_counts = {f'scene{i + 1}': 0 for i in range(len(scene_intervals))}
        transform_counts = {'M1': 0, 'M2': 0, 'M3': 0}
        ori_counts = 0

        for folder_name, timestamp_key in timestamps:
            # 仅 trans_M* 参与 transform / scene 计数
            if 'trans_M1' in folder_name:
                transform_counts['M1'] += 1
                do_scene = True
            elif 'trans_M2' in folder_name:
                transform_counts['M2'] += 1
                do_scene = True
            elif 'trans_M3' in folder_name:
                transform_counts['M3'] += 1
                do_scene = True
            elif 'ori_train' in folder_name:
                ori_counts += 1
                do_scene = False
            else:
                do_scene = False  # 其它目录不计
 
            if do_scene:
                for i, (start, end) in enumerate(scene_intervals):
                    if start <= int(timestamp_key) < end:
                        scene_counts[f'scene{i + 1}'] += 1
                        break
 
        distribution[method_name] = {
            'total_mutated': len(timestamps),
            'scene_counts_mutated': scene_counts,
            'transform_counts': transform_counts,
            'ori_counts': ori_counts,
        }

    # 统计原始数据分布（只统计一次）
    ori_scene_counts = None
    if ori_selected_timestamps is not None:
        ori_scene_counts = {f'scene{i + 1}': 0 for i in range(len(scene_intervals))}
        for frame_id in ori_selected_timestamps:
            for i, (start, end) in enumerate(scene_intervals):
                if start <= int(frame_id) < end:
                    ori_scene_counts[f'scene{i + 1}'] += 1
                    break



    # 保存统计
    statistics_dir = f'{save_dir}/selection_distribution/{scale}/{model}'
    os.makedirs(statistics_dir, exist_ok=True)
 
    with open(f'{statistics_dir}/selection_distribution.json', 'w') as f:
        json.dump({
            'methods': distribution,
            'ori_data': {
                'total_ori': len(ori_selected_timestamps) if ori_selected_timestamps else 0,
                'scene_counts_ori': ori_scene_counts if ori_scene_counts else {}
            }
        }, f, indent=4)
 
    report_path = f'{statistics_dir}/selection_distribution.txt'
    with open(report_path, 'w') as f:
        f.write(f'Selection distribution report\n')
        f.write(f'Model: {model}, Scale: {scale}\n\n')
 
        for method_name, md in distribution.items():
            f.write(f'[{method_name}]\n')
            f.write(f"  Mutated frames: {md['total_mutated']}\n")
            f.write('  Scene (mutated):\n')
            for k, v in md['scene_counts_mutated'].items():
                f.write(f'    {k}: {v}\n')
            f.write('  Transform mode:\n')
            for k, v in md['transform_counts'].items():
                f.write(f'    {k}: {v}\n')
            f.write('\n')
 
        if ori_selected_timestamps is not None:
            f.write(f'[Original Data (shared by all methods)]\n')
            f.write(f"  Total: {len(ori_selected_timestamps)}\n")
            f.write('  Scene distribution:\n')
            for k, v in ori_scene_counts.items():
                f.write(f'    {k}: {v}\n')
            f.write(f"  From ori_train: {md['ori_counts']}\n")
            f.write('\n')
 
    print(f'Selection distribution saved to {report_path}')


def select_scene_scores(result_stat, score_list, scene_intervals, scale):
    selected_list = []

    for interval in scene_intervals:
        scene_indices = []
        timestamp_list = result_stat['timestamp']

        for i, timestamp in enumerate(timestamp_list):
            if interval[0] <= int(timestamp[1]) < interval[1]:
                scene_indices.append(i)

        if scene_indices:
            sorted_scene_indices = sorted(scene_indices, key=lambda i: score_list[i], reverse=True)

            select_count = int(len(sorted_scene_indices) * scale)

            selected_list.extend(sorted_scene_indices[:select_count])

    return selected_list


def get_part_list_stat(total_state, part_list, part_state):
    for i in part_list:
        part_state[0.5]['tp'] += total_state[0.5]['tp'][i]
        part_state[0.5]['fp'] += total_state[0.5]['fp'][i]
        part_state[0.5]['gt'] += total_state[0.5]['gt'][i]
        part_state['occ_error'] += total_state['occ_error'][i]
        part_state['dis_error'] += total_state['dis_error'][i]
        part_state['total_occ'] += total_state['total_occ'][i]
        part_state['total_dis'] += total_state['total_dis'][i]
        if 'timestamp' in part_state:
            part_state['timestamp'].append(total_state['timestamp'][i])


def save_selected_data_and_label(timestamps, dataset_dir, save_dir, method):
    for i, timestamp in enumerate(timestamps):
        ego_pcd_path = f'{dataset_dir}/{timestamp[0]}/0/{timestamp[1]}.pcd'
        cp_pcd_path = f'{dataset_dir}/{timestamp[0]}/1/{timestamp[1]}.pcd'
        ego_label_path = f'{dataset_dir}/{timestamp[0]}/0/{timestamp[1]}.yaml'
        cp_label_path = f'{dataset_dir}/{timestamp[0]}/1/{timestamp[1]}.yaml'

        save_ego_folder = f'{save_dir}/{method}/0'
        save_cp_folder = f'{save_dir}/{method}/1'

        if not os.path.exists(save_ego_folder):
            os.makedirs(save_ego_folder)

        if not os.path.exists(save_cp_folder):
            os.makedirs(save_cp_folder)

        save_ego_pcd_path = f'{save_ego_folder}/{i:06d}.pcd'
        save_cp_pcd_path = f'{save_cp_folder}/{i:06d}.pcd'
        save_ego_label_path = f'{save_ego_folder}/{i:06d}.yaml'
        save_cp_label_path = f'{save_cp_folder}/{i:06d}.yaml'

        shutil.copy(ego_pcd_path, save_ego_pcd_path)
        shutil.copy(cp_pcd_path, save_cp_pcd_path)
        shutil.copy(ego_label_path, save_ego_label_path)
        shutil.copy(cp_label_path, save_cp_label_path)

def save_top_v2x_gen_data(timestamps, scores, dataset_dir, save_dir, select_num=996):
    if len(timestamps) != len(scores):
        raise ValueError(f'timestamps length {len(timestamps)} != scores length {len(scores)}')

    # final_save_dir = f'{save_dir}/rq3_selected_{select_num}/selected_data'
    final_save_dir = save_dir
    os.makedirs(final_save_dir, exist_ok=True)

    # 场景间隔定义
    intervals = [(0, 147), (147, 261), (261, 405), (405, 603), (603, 783),
                 (783, 1093), (1093, 1397), (1397, 1618), (1618, 1993)]

    selected_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:select_num]

    # 统计信息初始化
    transform_stats = {'M1': [], 'M2': [], 'M3': []}
    scene_stats = {f'Scene_{i+1}': [] for i in range(len(intervals))}
    detailed_frames = []

    # 分析选中的帧
    for original_idx in selected_indices:
        timestamp = timestamps[original_idx]
        score = scores[original_idx]
        folder_name = timestamp[0]
        timestamp_key = timestamp[1]
        
        # 提取变换模式
        transform_mode = None
        if 'M1' in folder_name:
            transform_mode = 'M1'
        elif 'M2' in folder_name:
            transform_mode = 'M2'
        elif 'M3' in folder_name:
            transform_mode = 'M3'
        
        # 确定场景
        scene_id = None
        for i, (start, end) in enumerate(intervals):
            if start <= int(timestamp_key) < end:
                scene_id = f'Scene_{i+1}'
                break
        
        # 记录统计信息
        if transform_mode:
            transform_stats[transform_mode].append({
                'index': original_idx,
                'timestamp': timestamp,
                'score': score
            })
        
        if scene_id:
            scene_stats[scene_id].append({
                'index': original_idx,
                'timestamp': timestamp,
                'score': score,
                'transform_mode': transform_mode
            })
        
        # 记录详细信息
        detailed_frames.append({
            'selected_rank': len(detailed_frames) + 1,
            'original_index': original_idx,
            'timestamp': timestamp,
            'score': score,
            'transform_mode': transform_mode,
            'scene_id': scene_id
        })
 
    # 保存统计信息
    save_selection_statistics(final_save_dir, transform_stats, scene_stats, detailed_frames, select_num)
    
    
    save_ego_folder = f'{final_save_dir}/0'
    save_cp_folder = f'{final_save_dir}/1'

    if not os.path.exists(save_ego_folder):
        os.makedirs(save_ego_folder)
    if not os.path.exists(save_cp_folder):
        os.makedirs(save_cp_folder)


    for new_idx, original_idx in enumerate(selected_indices):
        timestamp = timestamps[original_idx]

        ego_pcd_path = f'{dataset_dir}/{timestamp[0]}/0/{timestamp[1]}.pcd'
        cp_pcd_path = f'{dataset_dir}/{timestamp[0]}/1/{timestamp[1]}.pcd'
        ego_label_path = f'{dataset_dir}/{timestamp[0]}/0/{timestamp[1]}.yaml'
        cp_label_path = f'{dataset_dir}/{timestamp[0]}/1/{timestamp[1]}.yaml'

        save_ego_pcd_path = f'{save_ego_folder}/{new_idx:06d}.pcd'
        save_cp_pcd_path = f'{save_cp_folder}/{new_idx:06d}.pcd'
        save_ego_label_path = f'{save_ego_folder}/{new_idx:06d}.yaml'
        save_cp_label_path = f'{save_cp_folder}/{new_idx:06d}.yaml'

        shutil.copy(ego_pcd_path, save_ego_pcd_path)
        shutil.copy(cp_pcd_path, save_cp_pcd_path)
        shutil.copy(ego_label_path, save_ego_label_path)
        shutil.copy(cp_label_path, save_cp_label_path)

    print(f'Saved top {len(selected_indices)} V2X-Gen frames to {final_save_dir}')
    print(f'Statistics saved to {final_save_dir}/selection_statistics.json')

def save_selected_ori_data(ori_dataset_dir, save_dir, method, selected_timestamps):
    """
    保存预挑选的原始数据
    """
    save_ego_folder = f'{save_dir}/{method}/0'
    save_cp_folder = f'{save_dir}/{method}/1'

    if not os.path.exists(save_ego_folder):
        os.makedirs(save_ego_folder)
    if not os.path.exists(save_cp_folder):
        os.makedirs(save_cp_folder)

    existing_indices = [
        int(os.path.splitext(file_name)[0])
        for file_name in os.listdir(save_ego_folder)
        if file_name.endswith('.yaml') and os.path.splitext(file_name)[0].isdigit()
    ]
    start_idx = max(existing_indices) + 1 if len(existing_indices) > 0 else 0

    for i, timestamp in enumerate(selected_timestamps):
        save_idx = start_idx + i
        frame_id = timestamp

        ego_pcd_path = f'{ori_dataset_dir}/0/{frame_id}.pcd'
        cp_pcd_path = f'{ori_dataset_dir}/1/{frame_id}.pcd'
        ego_label_path = f'{ori_dataset_dir}/0/{frame_id}.yaml'
        cp_label_path = f'{ori_dataset_dir}/1/{frame_id}.yaml'

        save_ego_pcd_path = f'{save_ego_folder}/{save_idx:06d}.pcd'
        save_cp_pcd_path = f'{save_cp_folder}/{save_idx:06d}.pcd'
        save_ego_label_path = f'{save_ego_folder}/{save_idx:06d}.yaml'
        save_cp_label_path = f'{save_cp_folder}/{save_idx:06d}.yaml'

        shutil.copy(ego_pcd_path, save_ego_pcd_path)
        shutil.copy(cp_pcd_path, save_cp_pcd_path)
        shutil.copy(ego_label_path, save_ego_label_path)
        shutil.copy(cp_label_path, save_cp_label_path)

# def select_original_data(ori_dataset_dir, ori_scale=0.1, split_scene=False, scene_intervals=None):
def select_original_data(ori_dataset_dir, ori_num=896, split_scene=False, scene_intervals=None):
    """
    统一挑选原始数据
    
    Args:
        ori_dataset_dir: 原始数据目录
        ori_scale: 挑选比例
        split_scene: 是否按场景挑选
        scene_intervals: 场景时间戳区间列表
    
    Returns:
        selected_timestamps: 挑选的帧ID列表
    """
    ego_dir = f'{ori_dataset_dir}/0'
    
    # 获取所有帧ID
    frame_ids = sorted([
        os.path.splitext(file_name)[0]
        for file_name in os.listdir(ego_dir)
        if file_name.endswith('.yaml')
    ])
    
    if not split_scene:
        # 随机挑选
        # select_count = int(len(frame_ids) * ori_scale)
        select_count = ori_num
        selected_timestamps = sorted(random.sample(frame_ids, select_count))
    else:
        # 按场景挑选
        if scene_intervals is None:
            # 使用默认场景区间
            scene_intervals = [(0, 147), (147, 261), (261, 405), (405, 603), (603, 783),
                              (783, 1093), (1093, 1397), (1397, 1618), (1618, 1993)]
        
        selected_timestamps = []
        for start, end in scene_intervals:
            scene_frame_ids = [frame_id for frame_id in frame_ids if start <= int(frame_id) < end]
            if scene_frame_ids:
                # select_count = int(len(scene_frame_ids) * ori_scale)
                select_count = ori_num
                scene_selected = random.sample(scene_frame_ids, select_count)
                selected_timestamps.extend(scene_selected)
        
        selected_timestamps = sorted(selected_timestamps)
    
    return selected_timestamps

def v2x_select_eval(result_stat, model_dir, method):
    occ_error = result_stat['occ_error']
    long_dis_error = result_stat['dis_error']

    # occ_error_rate = occ_error / result_stat['total_occ']
    # long_dis_error_rate = long_dis_error / result_stat['total_dis']
    if result_stat['total_occ'] != 0:
        occ_error_rate = occ_error / result_stat['total_occ']
    else:
        occ_error_rate = 0.0
        
    if result_stat['total_dis'] != 0:
        long_dis_error_rate = long_dis_error / result_stat['total_dis']
    else:
        long_dis_error_rate = 0.0

    print(f"method = {method}, model = {model_dir}")
    print(f"ap_50 = {eval_final_results(result_stat, model_dir)}\n"
          f"occ error = {occ_error}, occ error rate = {occ_error_rate}\n"
          f"long dis error = {long_dis_error}, long dis error rate = {long_dis_error_rate}")


def v2x_eval_result(result_stat, model_dir, method):
    total_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0},
                         'occ_error': 0,
                         'dis_error': 0,
                         'total_occ': 0,
                         'total_dis': 0}
    num = len(result_stat[0.5]['tp'])
    get_part_list_stat(result_stat, range(num), total_result_stat)

    occ_error = total_result_stat['occ_error']
    long_dis_error = total_result_stat['dis_error']

    # occ_error_rate = occ_error / total_result_stat['total_occ']
    # long_dis_error_rate = long_dis_error / total_result_stat['total_dis']
    if total_result_stat['total_occ'] != 0:
        occ_error_rate = occ_error / total_result_stat['total_occ']
    else:
        occ_error_rate = 0.0

    if total_result_stat['total_dis'] != 0:
        long_dis_error_rate = long_dis_error / total_result_stat['total_dis']
    else:
        long_dis_error_rate = 0.0

    print("------------------------------------------------------")
    print(f"method = {method}, model = {model_dir}")
    print("------------------------------------------------------")
    print(f"ap_50 = {eval_final_results(total_result_stat, model_dir)}\n"
          f"occ error = {occ_error}, occ error rate = {occ_error_rate}\n"
          f"long dis error = {long_dis_error}, long dis error rate = {long_dis_error_rate}")
    print("------------------------------------------------------")

def save_selection_statistics(save_dir, transform_stats, scene_stats, detailed_frames, select_num):
    """
    保存筛选统计信息
    """
    import json
    
    # 准备统计数据
    statistics = {
        'selection_info': {
            'total_selected': select_num,
            'selection_method': 'V2XGen_fitness_score'
        },
        'transform_mode_distribution': {
            mode: {
                'count': len(frames),
                'frames': frames
            } for mode, frames in transform_stats.items()
        },
        'scene_distribution': {
            scene: {
                'count': len(frames),
                'frames': frames
            } for scene, frames in scene_stats.items()
        },
        'detailed_frame_list': detailed_frames
    }
    
    # 保存为JSON文件
    stats_file = os.path.join(save_dir, 'selection_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    # 生成可读的统计报告
    generate_readable_report(save_dir, statistics)
    
    # 打印简要统计
    print_selection_summary(statistics)

def generate_readable_report(save_dir, statistics):
    """
    生成可读的统计报告
    """
    report_file = os.path.join(save_dir, 'selection_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("V2XGen 筛选统计报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本信息
        f.write(f"总筛选帧数: {statistics['selection_info']['total_selected']}\n")
        f.write(f"筛选方法: {statistics['selection_info']['selection_method']}\n\n")
        
        # 变换模式分布
        f.write("变换模式分布:\n")
        f.write("-" * 30 + "\n")
        for mode, info in statistics['transform_mode_distribution'].items():
            f.write(f"{mode}: {info['count']} 帧\n")
        f.write("\n")
        
        # 场景分布
        f.write("场景分布:\n")
        f.write("-" * 30 + "\n")
        for scene, info in statistics['scene_distribution'].items():
            f.write(f"{scene}: {info['count']} 帧\n")
        f.write("\n")
        
        # 详细帧列表
        f.write("详细帧列表:\n")
        f.write("-" * 30 + "\n")
        for frame in statistics['detailed_frame_list']:
            f.write(f"排名 {frame['selected_rank']}: "
                   f"索引 {frame['original_index']}, "
                   f"时间戳 {frame['timestamp']}, "
                   f"分数 {frame['score']:.4f}, "
                   f"模式 {frame['transform_mode']}, "
                   f"场景 {frame['scene_id']}\n")

def print_selection_summary(statistics):
    """
    打印简要统计信息
    """
    print("\n" + "=" * 60)
    print("V2XGen 筛选统计摘要")
    print("=" * 60)
    
    print(f"总筛选帧数: {statistics['selection_info']['total_selected']}")
    
    print("\n变换模式分布:")
    for mode, info in statistics['transform_mode_distribution'].items():
        percentage = (info['count'] / statistics['selection_info']['total_selected']) * 100
        print(f"  {mode}: {info['count']} 帧 ({percentage:.1f}%)")
    
    print("\n场景分布:")
    for scene, info in statistics['scene_distribution'].items():
        if info['count'] > 0:
            percentage = (info['count'] / statistics['selection_info']['total_selected']) * 100
            print(f"  {scene}: {info['count']} 帧 ({percentage:.1f}%)")
    
    print("=" * 60)
