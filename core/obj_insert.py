import os
import random
import math
import config
import numpy as np
import open3d as o3d
import core.occlusion_treatment as occ
import utils.common_utils as common
import utils.visual as vis  # visualize the result of insert transformation

from utils.v2x_object import V2XInfo
from logger import CLogger
from utils.common_utils import get_corners, center_system_transform, rz_degree_system_transform, pc_numpy_2_o3d
from core.lidar_simulation import lidar_simulation, lidar_intensity_convert
from build.mtest.utils.Utils_o3d import load_normalized_mesh_obj
from build.mtest.utils.Utils_common import get_geometric_info, change_3dbox, get_initial_box3d_in_bg
from build.mtest.core.occusion_handing.combine_pc import combine_single_pcd
from build.mtest.core.pose_estimulation.pose_generator import tranform_mesh_by_pose, generate_pose


def vehicle_insert(ego_info, cp_info, position, detection_flag=False, gt_flag=False,
                   gt_degree=0, gt_box=None, transformation="insert"):
    ego_success_flag, ego_mesh, ego_insert_info, ego_combined_pc = insert_obj(ego_info, position, detection_flag, gt_flag, gt_degree, gt_box)
    

    if ego_success_flag:
        gt_degree = ego_insert_info["rz_degree"]

    cp_rz_degree = rz_degree_system_transform(gt_degree, ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])
    # 利用 center_system_transform 将该位置同步到 CP 的坐标系
    cp_position = list(center_system_transform(position, ego_info.param['lidar_pose'], cp_info.param['lidar_pose']))[:2]

    if ego_success_flag:
        T_ego2cp = np.linalg.inv(cp_info.param["lidar_pose"]) @ ego_info.param["lidar_pose"]
        gt_mesh = ego_mesh.transform(T_ego2cp)
        cp_success_flag, cp_mesh, cp_insert_info, cp_combined_pc = insert_obj(cp_info, cp_position, False, True, cp_rz_degree, gt_mesh=gt_mesh)
        if not cp_success_flag:
            return False, -1, -1
    else:
        cp_success_flag, cp_mesh, cp_insert_info, cp_combined_pc = insert_obj(cp_info, cp_position, True, False)
        if cp_success_flag:
            # 从 CP 端获取插入信息
            cp_rz_degree = cp_insert_info["rz_degree"]
        
            # 坐标系转换：CP → Ego
            ego_rz_degree = rz_degree_system_transform(
                cp_rz_degree, 
                cp_info.param['lidar_pose'], 
                ego_info.param['lidar_pose']
            )
            T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
            ego_gt_mesh = cp_mesh.transform(T_cp2ego)
            ego_success_flag, ego_mesh, ego_insert_info, ego_combined_pc = insert_obj(ego_info, position, False, True, ego_rz_degree, gt_mesh=ego_gt_mesh)
            if not ego_success_flag:
                return False, -1, -1

    # set occlusion threshold of insert object 
    # 计算遮挡率（Occlusion Rate），如果遮挡太严重（超过 0.9），则判定插入失败
    occlusion_threshold = 0.9
    if ego_success_flag:
        for val in ego_insert_info["occ_rate"].values():
            if val >= occlusion_threshold:
                ego_success_flag = False
                break

    if cp_success_flag:
        for val in cp_insert_info["occ_rate"].values():
            if val >= occlusion_threshold:
                cp_success_flag = False
                break

    if not cp_success_flag and not ego_success_flag:
        CLogger.info("insert false!")
        return False, -1, -1

    # after this step, insert transformation success!
    # 调用 occ.filter_part_occlusion_bg 剔除被新车挡住的背景点，保证物理一致性。
    ego_id = -1
    cp_id = -1
    
    if ego_success_flag:
        if ego_combined_pc is not None:
            ego_info.pc = ego_combined_pc

        for car_id, val in ego_insert_info["occ_rate"].items():
            if val > 0:
                CLogger.info("------------- filter part occlusion vehicle points")
                occ.filter_part_occlusion_bg(ego_info, car_id)

        for car_id, val in ego_insert_info["occluded_vehicles"].items():
            if val == "full":
                CLogger.info("------------- delete full occlusion vehicle")
                ego_info.delete_vehicle_of_id(car_id)

        if transformation != "insert":
            ego_id = ego_info.update_param_for_insert(ego_insert_info["extent"], ego_insert_info["center"],
                                                      -ego_insert_info["rz_degree"], use_old_id=True)
        else:
            ego_id = ego_info.update_param_for_insert(ego_insert_info["extent"], ego_insert_info["center"],
                                                      -ego_insert_info["rz_degree"])

    if cp_success_flag:
        if cp_combined_pc is not None:
            cp_info.pc = cp_combined_pc

        for car_id, val in cp_insert_info["occ_rate"].items():
            if val > 0:
                CLogger.info("------------- filter part occlusion vehicle points")
                occ.filter_part_occlusion_bg(cp_info, car_id)

        for car_id, val in cp_insert_info["occluded_vehicles"].items():
            if val == "full":
                CLogger.info("------------- delete full occlusion vehicle")
                cp_info.delete_vehicle_of_id(car_id)

        if ego_success_flag:
            ass_id = ego_id
        else:
            ass_id = -1

        if transformation != "insert":
            cp_id = cp_info.update_param_for_insert(cp_insert_info["extent"], cp_insert_info["center"],
                                                    -cp_insert_info["rz_degree"], use_old_id=True, ass_id=ass_id)
        else:
            cp_id = cp_info.update_param_for_insert(cp_insert_info["extent"], cp_insert_info["center"],
                                                    -cp_insert_info["rz_degree"], ass_id=ass_id)

    CLogger.info(f"insert complete! insert vehicle at {position[:2]}!")

    # # visualize after insert transformation
    # if ego_success_flag:


    #     vis_corner = ego_info.vehicles_info[ego_id]['corner'] 
    #     vis_cp_corner = common.points_system_transform(vis_corner, ego_info.param['lidar_pose'], cp_info.param['lidar_pose']) # 点云转换到cp视角      
        
    #     if gt_box is not None and transformation in ["rotation", "translation","scaling"]:
    #         print(f"Visualizing {transformation}: Red=Original, White=New")
    #         vis.show_ego_and_cp_for_translation_new(ego_info, cp_info, ego_id,gt_box,vis_corner)
    #         vis.show_obj_for_translation_new(ego_info, gt_box, vis_corner)
    #         vis.show_obj_for_translation_new(cp_info, gt_box, vis_cp_corner)

    #     vis.show_ego_and_cp_with_id(ego_info, cp_info, ego_id, cp_id) # 在两端点云融合的基础上，显示特定 ID 的车辆
    #     vis.show_obj_with_car_id(ego_info, ego_id) # 在ego点云中突出显示特定 ID 的车辆
    #     vis.show_obj_with_car_id(cp_info, cp_id)   # 在cv 点云中突出显示特定 ID 的车辆
    #     # vis_corner = ego_info.vehicles_info[ego_id]['corner']
    #     vis.show_ego_and_cp_for_translation(ego_info, cp_info, ego_id, vis_corner) # 在融合的点云场景中，同时查看原始位置和变换后的位置
    #     vis.show_obj_for_translation(ego_info, ego_id, vis_corner) # 对比显示ego的原始 Box 和 平移后的 Box
    #     # vis_cp_corner = common.points_system_transform(vis_corner, ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])
    #     vis.show_obj_for_translation(cp_info, cp_id, vis_cp_corner) # 对比显示cv的原始 Box 和 平移后的 Box
    # elif transformation == "rotation":
    #     return False, -1, -1
    # else:
    #     return False

    # return ego id and cp id
    if transformation == "rotation":
        if ego_success_flag and cp_success_flag:
            return True, ego_id, cp_id
        elif ego_success_flag:
            return True, ego_id, -1
        else:
            return True, -1, cp_id

    return True


# V2XGen 核心生成逻辑
def insert_obj(v2x_info, position, detection_flag=False, gt_flag=False, gt_degree=0, gt_box=None, gt_mesh=None):
    if v2x_info.get_vehicles_nums() == 0:
        corners_lidar = None
    else:
        corners_lidar = get_corners(v2x_info.param)  # vehicles corners list

    pos_z = select_road_height(v2x_info.road_pc, position)
    if len(position) == 2:
        position.append(pos_z)
    else:
        position[2] = pos_z

    # CLoggerCLogger.info(f"insert vehicle at {position}!")

    if gt_flag and gt_mesh is not None:
        mesh_obj_initial = gt_mesh
    else:
        obj_filename = config.common_config.obj_filename
        assets_dir = config.common_config.obj_dir_path
        obj_car_dirs = os.listdir(config.common_config.obj_dir_path)
        obj_num = len(obj_car_dirs)
        objs_index = np.random.randint(1, obj_num)
        obj_mesh_path = os.path.join(assets_dir, obj_car_dirs[objs_index], obj_filename)
        mesh_obj_initial = load_normalized_mesh_obj(obj_mesh_path)

    if corners_lidar is not None:
        initial_boxes, objs_half_diagonal, objs_center = get_initial_box3d_in_bg(corners_lidar)

    if gt_flag and gt_box is not None:
        mesh_obj_initial = resize_mesh_to_box(mesh_obj_initial, gt_box)
        rz_degree = gt_degree
    elif gt_flag:
        rz_degree = gt_degree
    else:
        pos, rz_degree = generate_pose(mesh_obj_initial, v2x_info.road_pc, v2x_info.road_label)

    half_diagonal, center, half_height = get_geometric_info(mesh_obj_initial)
    position[2] += half_height

    if not v2x_info.is_ego and gt_mesh is not None:
        mesh_obj = gt_mesh
    else:
        mesh_obj = tranform_mesh_by_pose(mesh_obj_initial, position, rz_degree)
        # tranform_mesh_by_pose 根据给定的坐标和旋转角度（Yaw），将 3D 模型移动到指定位置

    if detection_flag:
        # is mesh on road
        onroad_flag = is_on_road(mesh_obj, v2x_info.road_pc, v2x_info.no_road_pc, position, threshold=1)
        if not onroad_flag:
            CLogger.info("insert position is not on road!")
            return False, None, None, None

        # collision detect
        barycenter_xy = mesh_obj.get_center()[:2]
        if corners_lidar is not None:
            success_flag = collision_detection(barycenter_xy, half_diagonal, objs_half_diagonal, objs_center,# 检查新车是否与场景中原有车辆 (corners_lidar) 发生重叠。
                                               len(initial_boxes))
            if not success_flag:
                CLogger.info("insert position collision with other vehicles!")
                return False, None, None, None

    box_inserted_o3d = mesh_obj.get_minimal_oriented_bounding_box()

    box, angle = change_3dbox(box_inserted_o3d)

    try:
        pcd_obj = lidar_simulation(mesh_obj) # 调用 lidar_simulation 模拟激光雷达射线打在 Mesh 上生成的点云
    except ValueError:
        print("vehicles lidar simulation failed!")
        return False, None, None, None
    pcd_obj = lidar_intensity_convert(common.pc_numpy_2_o3d(v2x_info.pc), pcd_obj)

    # half of l,w,h
    extent = box.extent / 2
    location = box.center.copy()

    occ_rate_dict = occ.occlusion_rate_calculate(pcd_obj, v2x_info, position[2])
    occluded_vehicles = occ.insert_occlusion_detect(mesh_obj, v2x_info, position[2]) # 碰撞检测

    insert_info = {
        "extent": extent,
        "center": location,
        "rz_degree": rz_degree,
        "occ_rate": occ_rate_dict,
        "occluded_vehicles": occluded_vehicles
    }

    combined_pc = combine_pcd(v2x_info, pcd_obj, mesh_obj, position[2])

    return True, mesh_obj, insert_info, combined_pc

# 将 3D Mesh 模型进行缩放，使其符合目标边界框（Bounding Box）的尺寸
def resize_mesh_to_box(mesh, target_box):
    mesh_min_bound = mesh.get_min_bound()
    mesh_max_bound = mesh.get_max_bound()

    box_min_bound = np.min(target_box, axis=0)
    box_max_bound = np.max(target_box, axis=0)

    mesh_size = mesh_max_bound - mesh_min_bound
    target_size = box_max_bound - box_min_bound

    scale_factors = target_size / mesh_size
    mesh.scale(np.min(scale_factors), center=mesh.get_center())

    return mesh


# 通过寻找最近的路面点云，自动确定车辆的 Z 轴高度
def select_road_height(road_pc, position):
    road_x, road_y = road_pc[:, 0], road_pc[:, 1]

    distances = np.sqrt((road_x - position[0]) ** 2 + (road_y - position[1]) ** 2)

    nearest_distance = np.argmin(distances)

    nearest_pt_height = road_pc[nearest_distance, 2]

    return nearest_pt_height


# 检查插入位置是否在路面（Road）上，防止车出现在绿化带、建筑或空中
def is_on_road(mesh_obj, road_pc, non_road_pc, position, threshold=0.1):
    box = mesh_obj.get_oriented_bounding_box()
    non_road_pcd = pc_numpy_2_o3d(non_road_pc)
    non_road_pcd_contained = non_road_pcd.crop(box)

    road_x, road_y = road_pc[:, 0], road_pc[:, 1]

    distances = np.sqrt((road_x - position[0]) ** 2 + (road_y - position[1]) ** 2)

    nearest_distance = np.argmin(distances)

    if len(non_road_pcd_contained.points) >= 3 or nearest_distance <= threshold:
        return False
    else:
        return True


# 检查新车是否与场景中原有的车辆发生空间重叠
def collision_detection(xy, half_diagonal, objs_half_diagonal, objs_center, initial_box_num):
    occlusion_flag = config.not_behind_initial_obj and is_occlusion_initial_obj(xy, half_diagonal,
                                                                                objs_center[0:initial_box_num],
                                                                                objs_half_diagonal[0:initial_box_num])

    overlap_flag = is_3d_box_overlaped(objs_half_diagonal, objs_center, half_diagonal, xy)

    return (not overlap_flag) and (not occlusion_flag)


def is_3d_box_overlaped(diagonals, centers, half_diagonal, xy):
    if len(diagonals) == 0:
        return False
    for k in range(len(diagonals)):

        x_dis, y_dis = centers[k][0] - xy[0], centers[k][1] - xy[1]
        dis = math.sqrt(x_dis ** 2 + y_dis ** 2)

        length = diagonals[k] + half_diagonal

        if dis > length:
            pass
        else:

            return True
    return False


def is_occlusion_initial_obj(xy, half_diagonal, centers, diagonals):
    for k in range(len(diagonals)):
        y_dis = abs(centers[k][1] - xy[1])

        initial_obj_dis = math.sqrt(centers[k][0] ** 2 + centers[k][1] ** 2)
        insert_obj_dis = math.sqrt(xy[0] ** 2 + xy[1] ** 2)

        length = diagonals[k] + half_diagonal - 2

        if insert_obj_dis > initial_obj_dis and y_dis < length:
            return True

    return False

# 将新生成的车辆点云合并到背景点云中，并处理“影子区域”（即被新车挡住而消失的背景点）
def combine_pcd(v2x_info, obj, mesh,z_offset):
# def combine_pcd(v2x_info, obj, mesh):
    bg = pc_numpy_2_o3d(v2x_info.pc)
    print(bg.has_colors(), obj.has_colors())

    delete_mask, shadow_mesh = occ.get_delete_points_idx(mesh, bg)
    bg.points = o3d.utility.Vector3dVector(np.array(bg.points)[~delete_mask])  # 删掉遮挡区域点云
    bg.colors = o3d.utility.Vector3dVector(np.array(bg.colors)[~delete_mask])

    combined_pcd = bg + obj

    combined_pc = common.pcd_to_np(combined_pcd)

    return combined_pc


# 基准对照组入口，不进行复杂的物理遮挡剔除，主要用于对比实验（Baseline）
def base_insert(ego_info, cp_info, position, rz_degree):
    ego_success_flag, ego_mesh, ego_insert_info, ego_combined_pc = insert_obj_simple(ego_info, position, rz_degree)

    cp_rz_degree = rz_degree_system_transform(rz_degree, ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])
    cp_position = list(center_system_transform(position, ego_info.param['lidar_pose'], cp_info.param['lidar_pose']))
    T_ego2cp = np.linalg.inv(cp_info.param["lidar_pose"]) @ ego_info.param["lidar_pose"]
    gt_mesh = ego_mesh.transform(T_ego2cp)
    cp_success_flag, cp_mesh, cp_insert_info, cp_combined_pc = insert_obj_simple(cp_info, cp_position, cp_rz_degree, gt_mesh=gt_mesh)

    if not ego_success_flag and not cp_success_flag:
        return False

    ego_info.pc = ego_combined_pc
    ego_id = ego_info.update_param_for_insert(ego_insert_info["extent"], ego_insert_info["center"],
                                              -ego_insert_info["rz_degree"])

    cp_info.pc = cp_combined_pc
    cp_id = cp_info.update_param_for_insert(cp_insert_info["extent"], cp_insert_info["center"],
                                            -cp_insert_info["rz_degree"])

    # visualize after baseline insert
    # if ego_success_flag and cp_success_flag:
    #     vis.show_ego_and_cp_with_id(ego_info, cp_info, ego_id, cp_id)
    #     vis.show_obj_with_car_id(ego_info, ego_id)
    #     vis.show_obj_with_car_id(cp_info, cp_id)

    return True, ego_id, cp_id


# 直接对 Mesh 进行均匀采样（sample_points_uniformly），不模拟激光雷达的射线特性
def insert_obj_simple(v2x_info, position, rz_degree, gt_mesh=None):
    if gt_mesh is None:
        obj_filename = config.common_config.obj_filename
        assets_dir = config.common_config.obj_dir_path
        obj_car_dirs = os.listdir(config.common_config.obj_dir_path)
        obj_num = len(obj_car_dirs)
        objs_index = np.random.randint(1, obj_num)
        obj_mesh_path = os.path.join(assets_dir, obj_car_dirs[objs_index], obj_filename)
        mesh_obj_initial = load_normalized_mesh_obj(obj_mesh_path)

        half_diagonal, center, half_height = get_geometric_info(mesh_obj_initial)

        position[2] += half_height
        mesh_obj = tranform_mesh_by_pose(mesh_obj_initial, position, rz_degree)
    else:
        mesh_obj = gt_mesh

    box_inserted_o3d = mesh_obj.get_minimal_oriented_bounding_box()

    box, angle = change_3dbox(box_inserted_o3d)

    pcd_obj = mesh_obj.sample_points_uniformly(number_of_points=5000)

    # half of l,w,h
    extent = box.extent / 2
    location = box.center.copy()

    insert_info = {
        "extent": extent,
        "center": location,
        "rz_degree": rz_degree
    }

    combine_pc = base_combine_pcd(v2x_info, pcd_obj, box)

    return True, mesh_obj, insert_info, combine_pc


# 将新生成的车辆点云合并到背景点云中，并处理“影子区域”（即被新车挡住而消失的背景点）
def base_combine_pcd(v2x_info, obj, box):
    bg = pc_numpy_2_o3d(v2x_info.pc)

    infos = []
    delete_points_mask = None

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(box.get_box_points())

    triangles = [
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5]
    ]

    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    mesh.compute_vertex_normals()

    delete_mask, shadow_mesh = occ.get_delete_points_idx(mesh, bg)
    bg.points = o3d.utility.Vector3dVector(np.array(bg.points)[~delete_mask])  # 删掉遮挡区域点云

    occ.get_base_insert_shadowed_obj(obj, box)

    infos, bg, _ = combine_single_pcd(delete_points_mask, delete_mask, bg.points, obj, infos)

    return bg


# 使用 RANSAC 算法 拟合路面平面方程，确定可行驶区域
def get_road_plane_info(xyz):
    pcd = pc_numpy_2_o3d(xyz)

    distance_threshold = 0.01
    ransac_n = 3
    num_iterations = 1000

    road_range = [-200, 200, -200, 200, 0.35]

    plane, _ = pcd.segment_plane(distance_threshold=distance_threshold,
                                 ransac_n=ransac_n,
                                 num_iterations=num_iterations)
    [a, b, c, d] = plane

    A = plane[0]
    B = plane[1]
    C = plane[2]
    D = plane[3]

    idx = []
    min_x = min_y = math.inf
    max_x = max_y = -math.inf

    x_behind, x_front, y_left, y_right, abs_threshold = road_range
    for i in range(len(xyz)):
        setx, sety, setz = xyz[i]
        if x_behind <= setx <= x_front and y_left <= sety <= y_right:

            z = -1 * (A * setx + B * sety + D) / (C)
            if abs(setz - z) <= abs_threshold:
                idx.append(i)
                if min_x > setx:
                    min_x = setx
                if min_y > sety:
                    min_y = sety
                if max_x < setx:
                    max_x = setx
                if max_y < sety:
                    max_y = sety

    return idx, list(map(float, [min_x, max_x, min_y, max_y])), plane


# 在检测到的路面上随机生成一个合法的坐标点
def generate_base_insert_pos(bg_pc):
    road_pc_idx, road_pc_range, _ = get_road_plane_info(bg_pc)
    min_x, max_x, min_y, max_y = road_pc_range

    while True:
        random_idx = np.random.choice(road_pc_idx)
        selected_point_cloud = bg_pc[random_idx]
        if min_x <= selected_point_cloud[0] <= max_x and \
           min_y <= selected_point_cloud[1] <= max_y:
            return selected_point_cloud


if __name__ == '__main__':
    # baseline function test
    select_obj_list = []
    select_data_num = config.v2x_config.select_data_num

    index_list = list(range(1, select_data_num + 1))

    for bg_index in range(1, select_data_num + 1):
        index_list.remove(bg_index)
        ego_obj = V2XInfo(bg_index)
        cp_obj = V2XInfo(bg_index, is_ego=False)
        vehicle_num = len(ego_obj.param)

        # random get car index
        car_index = random.randint(0, vehicle_num - 1)

        pos = generate_base_insert_pos(ego_obj.pc)
        rz_degree = np.random.uniform(-180, 180)

        insert_flag = base_insert(ego_obj, cp_obj, pos, rz_degree)



