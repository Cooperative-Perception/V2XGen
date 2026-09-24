import argparse
import statistics
import sys
import os
import random

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.rq_eval.rq2_data_select import CooTest_method_result, V2X_Gen_method
from opencood.rq_eval.v2x_gen_utils import save_box_tensor, load_box_tensor, get_valid_param_dict, get_total_occ_and_dis

from utils.visual import visualize_individual_perspectives,calculate_max_iou_for_targets,filter_boxes_by_ids
import opencood.utils.common_utils as common_utils

def rq2_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='weight for occlusion-related score')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='weight for long-distance-related score')
    opt = parser.parse_args()
    return opt


def main():
    opt = rq2_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']

    if opt.fusion_method == 'nofusion':
        hypes = yaml_utils.load_yaml('model/late_fusion/config.yaml', None)
    else:
        hypes = yaml_utils.load_yaml("", opt)

    hypes['validate_dir'] = opt.dataset_dir

    print('Dataset Building')  # 加载数据集
    opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                     isSim=opt.isSim)

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {
        0.5: {'tp': [], 'fp': [], 'gt': []},
        0.7: {'tp': [], 'fp': [], 'gt': []},
        'occ_error': [],
        'dis_error': [],
        'total_occ': [],
        'total_dis': [],
        'timestamp': []
    }
    select_result_stat = {
        'v2x_gen': [],
        'cootest': [],
        'fop': [],
        'flp': []
    }

    total_fop = []
    total_flp = []

    # 用于全局统计新插入物体的变换质量
    global_transform_stats = {
        'total_inserted_objects': 0,  # 所有帧的新插入物体总数
        'valid_transforms': 0,  # 满足条件的变换总数
        'per_frame_stats': []  # 每帧的详细统计
    }

    if opt.fusion_method == 'nofusion':
        # nofusion method
        for i, batch_data in enumerate(data_loader):
            print('data idx =', i)

            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)

                det_box_tensor, det_score, gt_box_tensor, gt_object_ids = \
                    infrence_utils.inference_no_fusion(batch_data, model, opencood_dataset)
                ego_results = det_box_tensor, det_score, gt_box_tensor, gt_object_ids

                det_box_tensor_cp, det_score_cp, gt_box_tensor_cp, gt_object_ids_cp = \
                    infrence_utils.inference_no_fusion_cp(batch_data, model, opencood_dataset)
                cp_results = det_box_tensor_cp, det_score_cp, gt_box_tensor_cp, gt_object_ids_cp
                
                # 处理 CP 端无数据的情况
                if det_box_tensor_cp is None:
                     print(f"Warning: Frame {i} - CP has no valid data, setting empty results.")
                     det_box_tensor_cp = torch.zeros((0, 8, 3), device=device)
                     det_score_cp = torch.zeros((0,), device=device)
                     gt_box_tensor_cp = torch.zeros((0, 8, 3), device=device)
                     gt_object_ids_cp = []
                    

                ego_ins_ids = batch_data['ego']['inserted_ids']
                cp_ins_ids = batch_data['1']['inserted_ids']

                # 筛选新插入物体的 GT框
                ego_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor, gt_object_ids, ego_ins_ids)
                cp_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor_cp, gt_object_ids_cp, cp_ins_ids)

                # 在计算 IoU 之前添加
                print(f"CP GT boxes shape: {gt_box_tensor_cp.shape}")
                print(f"CP Pred boxes shape: {det_box_tensor_cp.shape}")
                print(f"CP inserted IDs: {cp_ins_ids}")
                print(f"CP GT object IDs: {gt_object_ids_cp}")

                # 检查 GT 框是否为空
                if gt_box_tensor_cp.shape[0] == 0:
                    print("Warning: CP GT boxes is empty!")
    
                # 检查预测框是否为空
                if det_box_tensor_cp.shape[0] == 0:
                    print("Warning: CP prediction boxes is empty!")

                # 计算 IoU - 针对每个新插入的 GT 物体，找到预测框中的最大 IoU
                ego_ious = calculate_max_iou_for_targets(det_box_tensor, ego_target_gt_boxes)
                cp_ious = calculate_max_iou_for_targets(det_box_tensor_cp, cp_target_gt_boxes)
                
                print(f"Ego inserted IDs: {ego_ins_ids}, CP inserted IDs: {cp_ins_ids}")
                print(f"Ego IoUs: {ego_ious}")
                print(f"CP IoUs: {cp_ious}")

                # ========== 利用 ass_id 建立 ego 和 cp 两端 ID 的映射关系 ==========
                # 直接从 batch_data 中获取 vehicles 信息（包含 ass_id）
                ego_vehicles = batch_data['ego'].get('vehicles', {})
                cp_vehicles = batch_data['1'].get('vehicles', {})
                
                print(f"\nDebug - Ego vehicles keys: {list(ego_vehicles.keys())}")
                print(f"Debug - CP vehicles keys: {list(cp_vehicles.keys())}")
                
                # 构建 cp_id 到 ass_id 的映射
                cp_id_to_ass_id = {}
                for cp_id, cp_vehicle in cp_vehicles.items():
                    ass_id = cp_vehicle.get('ass_id', -1)
                    cp_id_to_ass_id[cp_id] = ass_id
                
                print(f"Debug - CP ID to Ass_ID mapping: {cp_id_to_ass_id}")
                
                # ========== 使用 ass_id 进行 ID 合并 ==========
                # 核心逻辑：
                # 1. 如果 cp 端某车的 ass_id != -1，说明它对应 ego 端的 ass_id 那辆车
                # 2. 将这两端的车合并为同一个物理对象
                # 3. 对于 ass_id = -1 的 cp 端车辆，视为 cp 端独有的物体
                
                merged_inserted_objects = []  # 存储合并后的物体列表
                used_ego_ids = set()
                used_cp_ids = set()
                
                # 首先处理 ego 端的所有 inserted_ids
                for ego_id in ego_ins_ids:
                    merged_inserted_objects.append({
                        'unique_id': ego_id,
                        'ego_id': ego_id,
                        'cp_id': None,
                        'ego_exists': True,
                        'cp_exists': False
                    })
                    used_ego_ids.add(ego_id)
                
                # 然后处理 cp 端的物体，通过 ass_id 判断是否与 ego 端的物体关联
                for cp_id in cp_ins_ids:
                    if cp_id in used_cp_ids:
                        continue
                    
                    # 获取该 cp 车辆对应的 ego 车辆 ID（通过 ass_id）
                    ass_id = cp_id_to_ass_id.get(cp_id, -1)
                    
                    if ass_id != -1 and ass_id in used_ego_ids:
                        # 该 cp 车辆对应 ego 端的 ass_id 车辆，进行合并
                        for obj in merged_inserted_objects:
                            if obj['ego_id'] == ass_id:
                                obj['cp_exists'] = True
                                obj['cp_id'] = cp_id
                                used_cp_ids.add(cp_id)
                                print(f"  -> Merged: Ego ID {ass_id} with CP ID {cp_id} (ass_id={ass_id})")
                                break
                    else:
                        # cp 端独有的物体（ass_id=-1 或 ass_id 不在 ego 端）
                        merged_inserted_objects.append({
                            'unique_id': f"cp_{cp_id}",
                            'ego_id': None,
                            'cp_id': cp_id,
                            'ego_exists': False,
                            'cp_exists': True
                        })
                        used_cp_ids.add(cp_id)
                        print(f"  -> CP only: CP ID {cp_id} (ass_id={ass_id})")
                
                total_inserted_objects = len(merged_inserted_objects)
                print(f"\nMerged objects count: {total_inserted_objects}")
                for obj in merged_inserted_objects:
                    print(f"  Unique: {obj['unique_id']}, Ego: {obj['ego_id']}, CP: {obj['cp_id']}")
                
                if total_inserted_objects == 0:
                    print(f"\n=== Frame {i} Summary ===")
                    print(f"新插入物体总数：0 (跳过)")
                    print("========================\n")
                    continue

                valid_transform_count = 0
                transform_details = []  # 记录每个物体的详细变换信息

                # 遍历每个合并后的新插入物体
                for obj_info in merged_inserted_objects:
                    ego_id = obj_info['ego_id']
                    cp_id = obj_info['cp_id']
                    
                    # 获取该物体在 ego 端的 IoU
                    if obj_info['ego_exists'] and ego_id is not None:
                        if ego_id in ego_ins_ids:
                            ego_idx = ego_ins_ids.index(ego_id)
                            iou_e = ego_ious[ego_idx] if ego_idx < len(ego_ious) else 0.0
                        else:
                            iou_e = 0.0
                    else:
                        iou_e = 0.0  # ego 端没有该物体
                    
                    # 获取该物体在 cp 端的 IoU
                    if obj_info['cp_exists'] and cp_id is not None:
                        if cp_id in cp_ins_ids:
                            cp_idx = cp_ins_ids.index(cp_id)
                            iou_c = cp_ious[cp_idx] if cp_idx < len(cp_ious) else 0.0
                        else:
                            iou_c = 0.0
                    else:
                        iou_c = 0.0  # cp 端没有该物体
                    
                    # 判断条件：任意一端 IoU >= 0.5 即视为合理变换
                    is_valid = (iou_e >= 0.5 or iou_c >= 0.5)
                    if is_valid:
                        valid_transform_count += 1
                    
                    # 记录详细信息
                    transform_details.append({
                        'unique_id': obj_info['unique_id'],
                        'ego_id': ego_id,
                        'cp_id': cp_id,
                        'ego_iou': iou_e,
                        'cp_iou': iou_c,
                        'max_iou': max(iou_e, iou_c),
                        'is_valid': is_valid,
                        'ego_exists': obj_info['ego_exists'],
                        'cp_exists': obj_info['cp_exists']
                    })
                    
                    print(f"  Unique ID {obj_info['unique_id']}: "
                          f"Ego ID={ego_id} ({'✓' if obj_info['ego_exists'] else '✗'}), IoU={iou_e:.4f} | "
                          f"CP ID={cp_id} ({'✓' if obj_info['cp_exists'] else '✗'}), IoU={iou_c:.4f} | "
                          f"Max={max(iou_e, iou_c):.4f}, Valid={is_valid}")
                    # if is_valid == False:
                    #     print(undefined_variable)
                print(f"\n=== Frame {i} Summary ===")
                print(f"新插入物体总数：{total_inserted_objects}")
                print(f"满足条件 (Max(IoU) >= 0.5) 的数量：{valid_transform_count}")
                if total_inserted_objects > 0:
                    frame_valid_ratio = valid_transform_count / total_inserted_objects
                    print(f"合理变换比例：{frame_valid_ratio:.2%}")
                print("========================\n")

                # 累积到全局统计
                global_transform_stats['total_inserted_objects'] += total_inserted_objects
                global_transform_stats['valid_transforms'] += valid_transform_count
                global_transform_stats['per_frame_stats'].append({
                    'frame_idx': i,
                    'total_inserted': total_inserted_objects,
                    'valid_transforms': valid_transform_count,
                    'details': transform_details
                })

                # # 把rq3测试集根据适应度分数筛选指定数量的帧
                # fp, tp, gt, false_pred_ids = eval_utils.caluclate_tp_fp(
                #     det_box_tensor, det_score, gt_box_tensor, result_stat, 0.5,
                #     gt_object_ids=gt_object_ids.copy()
                # )
                # # 在 nofusion 循环中添加适应度计算
                # ego_v2x_gen_dict = batch_data['ego']['v2x_gen']
                # timestamp_key = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['timestamp']
                # folder_name = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['folder_name']
                # result_stat['timestamp'].append([folder_name, timestamp_key])
                
                # # 获取 CP 端的 v2x_gen 数据
                # cp_v2x_gen_dict = batch_data['1']['v2x_gen']
                # cp_v2x_gen_dict = get_valid_param_dict(cp_v2x_gen_dict, batch_data['1']['object_ids'])
                # ego_v2x_gen_dict = get_valid_param_dict(ego_v2x_gen_dict, batch_data['ego']['object_ids'])
                
                # # 计算适应度分数
                # gen_method_result, fop, flp = V2X_Gen_method(ego_v2x_gen_dict, cp_v2x_gen_dict,
                #                                         false_pred_ids, a=0.5, b=0.5)
                # select_result_stat['v2x_gen'].append(gen_method_result)
                # total_fop.append(fop)
                # total_flp.append(flp)

            # 预测结果分开保存
            det_save_path = "/home/zyc/code/V2XGen/rq2/rq2_det_box"
            det_save_path_cp = "/home/zyc/code/V2XGen/rq2/rq2_det_box_cp"
            if det_box_tensor is not None:
                save_box_tensor(det_box_tensor, det_score, i, det_save_path)
            if det_box_tensor_cp is not None:
                save_box_tensor(det_box_tensor_cp, det_score_cp, i, det_save_path_cp)
        
        # 输出全局统计结果
        print("\n" + "="*60)
        print("=== NOFUSION 全局变换质量评估 ===")
        print("="*60)
        print(f"总帧数：{len(global_transform_stats['per_frame_stats'])}")
        print(f"新插入物体总数：{global_transform_stats['total_inserted_objects']}")
        print(f"满足条件的变换数 (Max(IoU_ego, IoU_cp) >= 0.5): {global_transform_stats['valid_transforms']}")
        
        if global_transform_stats['total_inserted_objects'] > 0:
            global_valid_ratio = global_transform_stats['valid_transforms'] / global_transform_stats['total_inserted_objects']
            print(f"全局合理变换比例：{global_valid_ratio:.2%}")
            
            # 计算每帧的平均合理变换比例
            per_frame_ratios = []
            for frame_stat in global_transform_stats['per_frame_stats']:
                if frame_stat['total_inserted'] > 0:
                    ratio = frame_stat['valid_transforms'] / frame_stat['total_inserted']
                    per_frame_ratios.append(ratio)
            
            if len(per_frame_ratios) > 0:
                avg_per_frame_ratio = sum(per_frame_ratios) / len(per_frame_ratios)
                print(f"平均每帧合理变换比例：{avg_per_frame_ratio:.2%}")
                print(f"最高单帧合理变换比例：{max(per_frame_ratios):.2%}")
                print(f"最低单帧合理变换比例：{min(per_frame_ratios):.2%}")
        else:
            print("警告：没有检测到任何新插入的物体！")

        # # 保存适应度分数前996的帧
        # eval_utils.save_top_v2x_gen_data(
        #     timestamps=result_stat['timestamp'],
        #     scores=select_result_stat['v2x_gen'],
        #     dataset_dir=opt.dataset_dir,
        #     save_dir='/home/zyc/code/V2XGen/rq3',
        #     select_num=996,
        # )

        print("="*60 + "\n")
    else:
        # late/intermediate/early method
        for i, batch_data in enumerate(data_loader):
            print('data idx =', i)
            # if i > 100:
            #     break
            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)

                det_save_path = "/home/zyc/code/V2XGen/rq2/rq2_det_box"
                # 通过 load_box_tensor 加载之前保存的单车检测结果
                det_box_tensor, det_score = load_box_tensor(i, det_save_path)

                #根据指定的融合策略，调用 infrence_utils 中对应的函数获取融合后的预测结果。
                if opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                        infrence_utils.inference_late_fusion(batch_data,
                                                             model,
                                                             opencood_dataset)
                elif opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                        infrence_utils.inference_early_fusion(batch_data,
                                                              model,
                                                              opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                        infrence_utils.inference_intermediate_fusion(batch_data,
                                                                     model,
                                                                     opencood_dataset)
                else:
                    raise NotImplementedError('Only early, late and intermediate'
                                              'fusion is supported.')
                # overall calculating 计算标准感知指标
                fp, tp, gt, false_pred_ids = eval_utils.caluclate_tp_fp(pred_box_tensor,
                                                                        pred_score,
                                                                        gt_box_tensor,
                                                                        result_stat,
                                                                        0.5,
                                                                        gt_object_ids=gt_object_ids.copy())

                if gt != len(gt_object_ids):
                    sys.exit()

                ego_v2x_gen_dict = batch_data['ego']['v2x_gen']

                timestamp_key = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['timestamp']
                folder_name = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['folder_name']
                timestamp = [folder_name, timestamp_key]
                result_stat['timestamp'].append(timestamp)
                print(timestamp)

                if opt.fusion_method == 'late':
                    cp_v2x_gen_dict = batch_data['1']['v2x_gen']
                    cp_v2x_gen_dict = get_valid_param_dict(cp_v2x_gen_dict, batch_data['1']['object_ids'])
                else:
                    cp_v2x_gen_dict = batch_data['ego']['v2x_gen_cp']

                ego_v2x_gen_dict = get_valid_param_dict(ego_v2x_gen_dict, batch_data['ego']['object_ids'])

                occ_error, total_occ = eval_utils.get_occ_error(ego_v2x_gen_dict, cp_v2x_gen_dict,
                                                                false_pred_ids)
                dis_error, total_dis = eval_utils.get_long_distance_error(ego_v2x_gen_dict, cp_v2x_gen_dict,
                                                                          false_pred_ids, 50)

                result_stat['occ_error'].append(occ_error)
                result_stat['dis_error'].append(dis_error)
                result_stat['total_occ'].append(total_occ)
                result_stat['total_dis'].append(total_dis)

                cootest_method_result = CooTest_method_result(det_box_tensor, det_score, pred_box_tensor)
                gen_method_result, fop, flp = V2X_Gen_method(ego_v2x_gen_dict, cp_v2x_gen_dict,
                                                             false_pred_ids,
                                                             a=opt.alpha,
                                                             b=opt.beta)
                select_result_stat['v2x_gen'].append(gen_method_result)
                select_result_stat['fop'].append(fop)
                select_result_stat['flp'].append(flp)
                total_flp.append(flp)
                total_fop.append(fop)
                # print(gen_method_result)
                select_result_stat['cootest'].append(cootest_method_result)
                print('cootest param =', cootest_method_result, ', v2x gen param =', gen_method_result)
                print('occ error =', occ_error, ', dis error =', dis_error)

                # # if opt.fusion_method == 'late':
                # #     pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                # #             infrence_utils.inference_late_fusion(batch_data, model, opencood_dataset)
                # # else:
                #     # if len(tp) < gt: #只展示漏检的帧
                #     # print(f"检测到漏检（TP:{tp}/GT:{gt}），弹出窗口分析原因...")
                #     # if i % 50 == 0:
                # visual.visualize_and_eval_post_operation(
                #             v2x_info=batch_data['ego'], 
                #             pred_box_tensor=pred_box_tensor, 
                #             pred_scores=pred_score, 
                #             gt_box_tensor=gt_box_tensor, 
                #             iou_threshold=0.5, 
                #             is_ego=True, 
                #             window_name=f"Frame {i} - Ego Fusion Perspective"
                #         )
    # result
    if opt.fusion_method == 'nofusion':
        eval_utils.eval_final_results(result_stat,
                                      opt.model_dir)
    else:
        # # 统一挑选原始数据
        # ori_dataset_dir = '/home/zyc/code/V2XGen/rq2/rq2_gen/ori_train'  # 原始训练数据集
        # # ori_dataset_dir = '/home/zyc/code/V2XGen/rq2/rq2_gen/ori_data'   # T1原始数据
        # # ori_scale = 0.1
        # ori_num = 996
        # split_scene = False  # 设置为 True 则分场景挑选，False 则随机挑选

        # model_name = opt.model_dir.split('/')[-1]
        # random.seed(hash(model_name) % (2**32))

        # ori_selected_timestamps = eval_utils.select_original_data(
        #     ori_dataset_dir, 
        #     # ori_scale=ori_scale, 
        #     ori_num=ori_num,
        #     split_scene=split_scene
        # )

        # eval_utils.method_eval_result(select_result_stat, result_stat, opt.model_dir, 0.1, True, opt.dataset_dir,
        #                               '/home/zyc/code/V2XGen/rq2/rq2_select', opt.model_dir.split('/')[-1])
        # eval_utils.method_eval_result(select_result_stat, result_stat, opt.model_dir, 0.15, True, opt.dataset_dir,
        #                               '/home/zyc/code/V2XGen/rq2/rq2_select', opt.model_dir.split('/')[-1])

        eval_utils.method_eval_result(select_result_stat,
                                      result_stat,
                                      opt.model_dir, 
                                      0.15, 
                                      True, 
                                      opt.dataset_dir,
                                      '/home/zyc/code/V2XGen/rq2/rq2_select', 
                                      opt.model_dir.split('/')[-1])

        # model_name = opt.model_dir.split('/')[-1]
        # save_dir = f'/home/zyc/code/V2XGen/rq3/model_select/{model_name}/996'
    
        # eval_utils.save_top_v2x_gen_data(
        #     timestamps=result_stat['timestamp'],
        #     scores=select_result_stat['v2x_gen'],
        #     dataset_dir=opt.dataset_dir,
        #     save_dir=save_dir,
        #     select_num=996,
        # )

        print("total FOP =", sum(total_fop), "total FLP =", sum(total_flp))
        print("avg FOP =", sum(total_fop) / len(total_fop))
        print("avg FLP =", sum(total_flp) / len(total_flp))
        print("mid FOP =", statistics.median(total_fop))
        print("mid FLP =", statistics.median(total_flp))


if __name__ == '__main__':
    main()
