import argparse
import sys
import os
import shutil
import random

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from utils.visual import visualize_individual_perspectives,calculate_max_iou_for_targets,filter_boxes_by_ids


# 导入变换相关模块
sys.path.append('/home/zyc/code/V2XGen/rq2')
import obj_transformation as trans
from config.config import Config
from utils.v2x_object import V2XInfo


def rq2_valid_parser():
    parser = argparse.ArgumentParser(description="Generate valid frames for RQ2/RQ3")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Valid output dataset dir')
    parser.add_argument('--temp_dir', type=str, default=None,
                        help='Temporary generation dir')
    parser.add_argument('--dataset_type', type=str, default='rq2', choices=['rq2', 'rq3'],
                        help='Dataset type for regeneration')
    parser.add_argument('--max_attempts', type=int, default=10,
                        help='Maximum attempts to generate a valid frame')
    opt = parser.parse_args()

    if opt.output_dir is None:
        if opt.dataset_type == 'rq3':
            opt.output_dir = '/home/zyc/code/V2XGen/rq3/rq3_test_valid'
        else:
            opt.output_dir = '/home/zyc/code/V2XGen/rq_eval_valid/rq2_gen'

    if opt.temp_dir is None:
        opt.temp_dir = '/home/zyc/code/V2XGen/rq_eval/temp_gen'
    return opt


def evaluate_frame_validity(model, opencood_dataset, batch_data, device):
    """评估单帧的有效性"""
    with torch.no_grad():
        batch_data = train_utils.to_device(batch_data, device)

        det_box_tensor, det_score, gt_box_tensor, gt_object_ids = \
            infrence_utils.inference_no_fusion(batch_data, model, opencood_dataset)

        det_box_tensor_cp, det_score_cp, gt_box_tensor_cp, gt_object_ids_cp = \
            infrence_utils.inference_no_fusion_cp(batch_data, model, opencood_dataset)
        
        # 处理 CP 端无数据的情况
        if det_box_tensor_cp is None:
            det_box_tensor_cp = torch.zeros((0, 8, 3), device=device)
            det_score_cp = torch.zeros((0,), device=device)
            gt_box_tensor_cp = torch.zeros((0, 8, 3), device=device)
            gt_object_ids_cp = []

        ego_ins_ids = batch_data['ego']['inserted_ids']
        cp_ins_ids = batch_data['1']['inserted_ids']

        # 筛选新插入物体的 GT框
        ego_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor, gt_object_ids, ego_ins_ids)
        cp_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor_cp, gt_object_ids_cp, cp_ins_ids)

        # 计算 IoU - 针对每个新插入的 GT 物体，找到预测框中的最大 IoU
        ego_ious = calculate_max_iou_for_targets(det_box_tensor, ego_target_gt_boxes)
        cp_ious = calculate_max_iou_for_targets(det_box_tensor_cp, cp_target_gt_boxes)

        # 利用 ass_id 建立 ego 和 cp 两端 ID 的映射关系
        ego_vehicles = batch_data['ego'].get('vehicles', {})
        cp_vehicles = batch_data['1'].get('vehicles', {})
        
        # 构建 cp_id 到 ass_id 的映射
        cp_id_to_ass_id = {}
        for cp_id, cp_vehicle in cp_vehicles.items():
            ass_id = cp_vehicle.get('ass_id', -1)
            cp_id_to_ass_id[cp_id] = ass_id

        # 使用 ass_id 进行 ID 合并
        merged_inserted_objects = []
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
            
            ass_id = cp_id_to_ass_id.get(cp_id, -1)
            
            if ass_id != -1 and ass_id in used_ego_ids:
                # 该 cp 车辆对应 ego 端的 ass_id 车辆，进行合并
                for obj in merged_inserted_objects:
                    if obj['ego_id'] == ass_id:
                        obj['cp_exists'] = True
                        obj['cp_id'] = cp_id
                        used_cp_ids.add(cp_id)
                        break
            else:
                # cp 端独有的物体
                merged_inserted_objects.append({
                    'unique_id': f"cp_{cp_id}",
                    'ego_id': None,
                    'cp_id': cp_id,
                    'ego_exists': False,
                    'cp_exists': True
                })
                used_cp_ids.add(cp_id)

        total_inserted_objects = len(merged_inserted_objects)
        if total_inserted_objects == 0:
            # 当新插入物体为0时（如只执行delete操作），视为合理帧
            return True, 0, 0

        valid_transform_count = 0
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
                iou_e = 0.0
            
            # 获取该物体在 cp 端的 IoU
            if obj_info['cp_exists'] and cp_id is not None:
                if cp_id in cp_ins_ids:
                    cp_idx = cp_ins_ids.index(cp_id)
                    iou_c = cp_ious[cp_idx] if cp_idx < len(cp_ious) else 0.0
                else:
                    iou_c = 0.0
            else:
                iou_c = 0.0
            
            # 判断条件：任意一端 IoU >= 0.5 即视为合理变换
            is_valid = (iou_e >= 0.5 or iou_c >= 0.5)
            if is_valid:
                valid_transform_count += 1

        # 判断是否为有效帧（所有新插入物体都是Valid）
        is_valid_frame = (valid_transform_count == total_inserted_objects)
        return is_valid_frame, total_inserted_objects, valid_transform_count


def generate_and_validate_frame(
    bg_index,
    transformation_mode,
    model,
    opencood_dataset,
    device,
    max_attempts=50,
    dataset_type='rq2',
    temp_gen_dir='/home/zyc/code/V2XGen/rq_eval/temp_gen'
):
    """生成并验证一个有效的帧"""
    print(f"Generating and validating frame for bg_index {bg_index}, mode {transformation_mode}")
    
    # 创建临时配置用于生成数据
    dataset_config = Config(dataset=dataset_type)
    # 修改临时目录结构：直接在temp_gen_dir下创建变换模式目录
    os.makedirs(temp_gen_dir, exist_ok=True)
    
    for attempt in range(max_attempts):
        print(f"  Attempt {attempt + 1}/{max_attempts}")
        
        try:
            # 清理临时目录 - 现在清理整个temp_gen_dir下的所有内容
            if os.path.exists(temp_gen_dir):
                shutil.rmtree(temp_gen_dir)
            os.makedirs(temp_gen_dir, exist_ok=True)
            
            # 重新定义 temp_mode_dir
            temp_mode_dir = os.path.join(temp_gen_dir, transformation_mode)
            
            # 设置临时保存目录 - 直接保存到temp_gen_dir下
            dataset_config.v2x_dataset_saved_dir = temp_gen_dir
            
            # 创建新的V2XInfo对象
            ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
            cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)
            
            # 执行变换
            op_count = 0
            total_car_num = ego_info.get_vehicles_nums()
            selected_car_id = []
            OP_TIMES = int(transformation_mode.split('_M')[1])
            
            while op_count < OP_TIMES:
                transformation_list = ["insert", "delete", "translation", "scaling", "rotation"]
                transformation = random.choice(transformation_list)
                
                car_id = 0
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
                        op_count += 1
                        continue

                success_flag = False
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
                
                if success_flag:
                    op_count += 1
            
            # 标签补全
            trans.label_complete_for_ego(ego_info, cp_info)
            trans.label_complete_for_cp(ego_info, cp_info)

            # 保存到临时目录 - 传入正确的folder_name (transformation_mode)
            ego_info.save_data_and_label(transformation_mode, save_to_pcd=True)
            cp_info.save_data_and_label(transformation_mode, save_to_pcd=True)
            
            # 创建临时数据集配置来加载这个单帧
            temp_hypes = yaml_utils.load_yaml('model/late_fusion/config.yaml', None)
            temp_hypes['validate_dir'] = temp_gen_dir
            
            # 构建临时数据集（只包含这一帧）
            from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
            temp_dataset = LateFusionDataset(temp_hypes, visualize=True, train=False, isSim=True)
            
            if len(temp_dataset) == 0:
                print(f"    Temp dataset empty, retrying...")
                continue
                
            # 创建DataLoader
            temp_loader = DataLoader(temp_dataset,
                                   batch_size=1,
                                   num_workers=0,  # 避免多进程问题
                                   collate_fn=temp_dataset.collate_batch_test,
                                   shuffle=False,
                                   pin_memory=False,
                                   drop_last=False)
            
            # 获取batch数据
            temp_batch = next(iter(temp_loader))
            
            # 验证帧的有效性
            is_valid, total_objs, valid_objs = evaluate_frame_validity(model, temp_dataset, temp_batch, device)
            
            if is_valid and total_objs >= 0:
                print(f"    Successfully generated valid frame! Total objects: {total_objs}, Valid: {valid_objs}")
                # 返回临时目录中的数据路径和信息
                return temp_mode_dir, bg_index, total_objs, valid_objs
            else:
                print(f"    Generated frame is not valid. Total objects: {total_objs}, Valid: {valid_objs}")
                
        except Exception as e:
            import traceback
            print(f"    [ERROR] Attempt {attempt + 1} failed.")
            print(f"    Exception Type: {type(e).__name__}")
            print(f"    Exception Message: {str(e)}")
            print("    Full Traceback:")
            traceback.print_exc()
            continue
    
    print(f"Failed to generate valid frame for bg_index {bg_index} after {max_attempts} attempts")
    return None, None, 0, 0


def main():
    opt = rq2_valid_parser()
    
    # 加载模型
    hypes = yaml_utils.load_yaml('model/late_fusion/config.yaml', None)
    hypes['validate_dir'] = opt.dataset_dir
    
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, isSim=True)
    
    print('Creating Model')
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()
    
    # 创建目标保存目录
    valid_save_root = opt.output_dir
    if opt.dataset_type == 'rq3':
        modes = ['test_M1', 'test_M2', 'test_M3']
    else:
        modes = ['trans_M1', 'trans_M2', 'trans_M3']
    for mode in modes:
        mode_dir = os.path.join(valid_save_root, mode)
        os.makedirs(mode_dir, exist_ok=True)
        # 清空目录以确保干净开始
        for item in os.listdir(mode_dir):
            item_path = os.path.join(mode_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    
    data_loader = DataLoader(opencood_dataset,
                           batch_size=1,
                           num_workers=4,
                           collate_fn=opencood_dataset.collate_batch_test,
                           shuffle=False,
                           pin_memory=False,
                           drop_last=False)
    
    processed_frames = {mode: 0 for mode in modes}
    
    print(f"Processing {len(opencood_dataset)} frames...")
    
    for i, batch_data in enumerate(data_loader):
        print(f'Processing frame {i+1}/{len(opencood_dataset)}')
        
        with torch.no_grad():
            # 获取变换模式信息
            ego_v2x_gen_dict = batch_data['ego']['v2x_gen']
            if not ego_v2x_gen_dict:
                print(f"Frame {i}: No v2x_gen info, skipping")
                continue
                
            first_key = list(ego_v2x_gen_dict.keys())[0]
            folder_name = ego_v2x_gen_dict[first_key]['folder_name']
            timestamp = ego_v2x_gen_dict[first_key]['timestamp']
            
            # 从folder_name中提取变换模式
            mode_match = None
            for mode in modes:
                if mode in folder_name:
                    mode_match = mode
                    break
            
            if not mode_match:
                print(f"Frame {i}: Cannot determine transformation mode from {folder_name}, skipping")
                continue
            
            # 评估帧的有效性
            is_valid, total_objs, valid_objs = evaluate_frame_validity(model, opencood_dataset, batch_data, device)
            
            print(f"DEBUG: Frame {i} evaluation result - is_valid={is_valid}, total_objs={total_objs}, valid_objs={valid_objs}")
            
            if is_valid and total_objs >= 0:  # 修改条件：total_objs可以为0
                print(f"Frame {i} ({mode_match}) is already valid! Total objects: {total_objs}, Valid: {valid_objs}")
                
                # 直接复制原始数据到目标目录
                try:
                    # 获取原始数据路径
                    original_data_dir = opt.dataset_dir
                    bg_index_str = f"{int(timestamp):06d}" if isinstance(timestamp, (int, str)) else f"{i:06d}"
                    
                    # 源路径
                    src_ego_pcd = os.path.join(original_data_dir, mode_match, "0", f"{bg_index_str}.pcd")
                    src_ego_yaml = os.path.join(original_data_dir, mode_match, "0", f"{bg_index_str}.yaml")
                    src_cp_pcd = os.path.join(original_data_dir, mode_match, "1", f"{bg_index_str}.pcd")
                    src_cp_yaml = os.path.join(original_data_dir, mode_match, "1", f"{bg_index_str}.yaml")
                    
                    # 目标路径
                    target_mode_dir = os.path.join(valid_save_root, mode_match)
                    target_ego_dir = os.path.join(target_mode_dir, "0")
                    target_cp_dir = os.path.join(target_mode_dir, "1")
                    os.makedirs(target_ego_dir, exist_ok=True)
                    os.makedirs(target_cp_dir, exist_ok=True)
                    
                    # 复制文件
                    if os.path.exists(src_ego_pcd):
                        shutil.copy2(src_ego_pcd, os.path.join(target_ego_dir, f"{bg_index_str}.pcd"))
                    if os.path.exists(src_ego_yaml):
                        shutil.copy2(src_ego_yaml, os.path.join(target_ego_dir, f"{bg_index_str}.yaml"))
                    if os.path.exists(src_cp_pcd):
                        shutil.copy2(src_cp_pcd, os.path.join(target_cp_dir, f"{bg_index_str}.pcd"))
                    if os.path.exists(src_cp_yaml):
                        shutil.copy2(src_cp_yaml, os.path.join(target_cp_dir, f"{bg_index_str}.yaml"))
                    
                    processed_frames[mode_match] += 1
                    print(f"Copied valid frame {bg_index_str} to {mode_match} (count: {processed_frames[mode_match]})")
                    
                except Exception as e:
                    print(f"Error copying frame {i}: {e}")
                    # 如果复制失败，尝试重新生成并验证
                    print(f"Regenerating and validating frame {i} due to copy error...")
                    temp_dir, bg_idx, total_o, valid_o = generate_and_validate_frame(
                                                            int(timestamp),
                                                            mode_match,
                                                            model,
                                                            opencood_dataset,
                                                            device,
                                                            opt.max_attempts,
                                                            opt.dataset_type,
                                                            opt.temp_dir
                                                        )
                    if temp_dir is not None:
                        # 复制验证有效的数据
                        bg_index_str = f"{int(timestamp):06d}"
                        target_mode_dir = os.path.join(valid_save_root, mode_match)
                        target_ego_dir = os.path.join(target_mode_dir, "0")
                        target_cp_dir = os.path.join(target_mode_dir, "1")
                        os.makedirs(target_ego_dir, exist_ok=True)
                        os.makedirs(target_cp_dir, exist_ok=True)
                        
                        # 从临时目录复制
                        temp_ego_pcd = os.path.join(temp_dir, "0", f"{bg_index_str}.pcd")
                        temp_ego_yaml = os.path.join(temp_dir, "0", f"{bg_index_str}.yaml")
                        temp_cp_pcd = os.path.join(temp_dir, "1", f"{bg_index_str}.pcd")
                        temp_cp_yaml = os.path.join(temp_dir, "1", f"{bg_index_str}.yaml")
                        
                        if os.path.exists(temp_ego_pcd):
                            shutil.copy2(temp_ego_pcd, os.path.join(target_ego_dir, f"{bg_index_str}.pcd"))
                        if os.path.exists(temp_ego_yaml):
                            shutil.copy2(temp_ego_yaml, os.path.join(target_ego_dir, f"{bg_index_str}.yaml"))
                        if os.path.exists(temp_cp_pcd):
                            shutil.copy2(temp_cp_pcd, os.path.join(target_cp_dir, f"{bg_index_str}.pcd"))
                        if os.path.exists(temp_cp_yaml):
                            shutil.copy2(temp_cp_yaml, os.path.join(target_cp_dir, f"{bg_index_str}.yaml"))
                        
                        processed_frames[mode_match] += 1
                        print(f"Saved regenerated and validated frame {bg_index_str} to {mode_match}")
            
            else:
                print(f"Frame {i} ({mode_match}) is not valid. Total objects: {total_objs}, Valid: {valid_objs}")
                print(f"Regenerating and validating frame {i} until valid...")
                
                # 重新生成并验证有效帧
                try:
                    temp_dir, bg_idx, total_o, valid_o = generate_and_validate_frame(
                        int(timestamp), mode_match, model, opencood_dataset, device, opt.max_attempts, opt.dataset_type, opt.temp_dir)
                    
                    if temp_dir is not None:
                        # 保存验证有效的帧到目标目录
                        bg_index_str = f"{int(timestamp):06d}"
                        target_mode_dir = os.path.join(valid_save_root, mode_match)
                        target_ego_dir = os.path.join(target_mode_dir, "0")
                        target_cp_dir = os.path.join(target_mode_dir, "1")
                        os.makedirs(target_ego_dir, exist_ok=True)
                        os.makedirs(target_cp_dir, exist_ok=True)
                        
                        # 从临时目录复制
                        temp_ego_pcd = os.path.join(temp_dir, "0", f"{bg_index_str}.pcd")
                        temp_ego_yaml = os.path.join(temp_dir, "0", f"{bg_index_str}.yaml")
                        temp_cp_pcd = os.path.join(temp_dir, "1", f"{bg_index_str}.pcd")
                        temp_cp_yaml = os.path.join(temp_dir, "1", f"{bg_index_str}.yaml")
                        
                        if os.path.exists(temp_ego_pcd):
                            shutil.copy2(temp_ego_pcd, os.path.join(target_ego_dir, f"{bg_index_str}.pcd"))
                        if os.path.exists(temp_ego_yaml):
                            shutil.copy2(temp_ego_yaml, os.path.join(target_ego_dir, f"{bg_index_str}.yaml"))
                        if os.path.exists(temp_cp_pcd):
                            shutil.copy2(temp_cp_pcd, os.path.join(target_cp_dir, f"{bg_index_str}.pcd"))
                        if os.path.exists(temp_cp_yaml):
                            shutil.copy2(temp_cp_yaml, os.path.join(target_cp_dir, f"{bg_index_str}.yaml"))
                        
                        processed_frames[mode_match] += 1
                        print(f"Saved regenerated and validated frame {bg_index_str} to {mode_match}")
                    else:
                        print(f"Failed to regenerate valid frame for timestamp {timestamp}")
                        
                except Exception as e:
                    print(f"Error regenerating frame {i}: {e}")
    
    # 清理临时目录
    if os.path.exists(opt.temp_dir):
        shutil.rmtree(opt.temp_dir)
    
    print("\n=== Process Complete ===")
    print("Original data remains in:", opt.dataset_dir)
    print("Valid frames saved to:", valid_save_root)
    print("Processed frames per mode:")
    for mode, count in processed_frames.items():
        print(f"  {mode}: {count} frames")


if __name__ == '__main__':
    main()