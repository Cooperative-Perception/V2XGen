import sys
import config
import random
import numpy as np
import open3d as o3d
import core.occlusion_treatment as occ
import utils.common_utils as common
import utils.visual as vis
from logger import CLogger
from utils.v2x_object import V2XInfo
from utils.common_utils import pc_numpy_2_o3d, is_box_containing_position
from core.obj_insert import insert_obj
from core.occlusion_treatment import get_delete_points_idx
from core.lidar_simulation import lidar_simulation, lidar_intensity_convert

# 控制流函数，执行高级删除逻辑
def vehicle_delete(ego_info, cp_info, car_id):
    CLogger.info(f"background index = {ego_info.bg_index}, delete ego vehicle id = {car_id}")

    # visualize before delete
    corner = ego_info.vehicles_info[car_id]["corner"]
    # vis.show_obj_with_corner(ego_info, corner)
    # cp_corner = common.points_system_transform(corner, ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])

    # # cp_car_id = car_id # 或者通过 ass_id 匹配，取决于你的数据集映射逻辑
    # # if cp_car_id in cp_info.vehicles_info:
    # #     cp_corner = cp_info.vehicles_info[cp_car_id]["corner"]
    # # # 执行协作车视角的删除/处理逻辑...
    # # else:
    # #     CLogger.info(f"Vehicle {cp_car_id} not found in CP view, skipping CP deletion.")

    # # cp_corner = cp_info.vehicles_info[1]["corner"]
    # vis.show_obj_with_corner(cp_info, cp_corner)
    # vis.show_ego_and_cp_with_corner(cp_info, ego_info, cp_corner)

    # # # visualize before delete
    # #   corner = ego_info.vehicles_info[car_id]["corner"]
    # # # vis.show_obj_with_corner(ego_info, corner)
    # # # cp_corner = common.points_system_transform(corner, ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])
    # # # cp_corner = cp_info.vehicles_info[1]["corner"]
    # # # vis.show_obj_with_corner(cp_info, cp_corner)
    # # # vis.show_ego_and_cp_with_corner(cp_info, ego_info, cp_corner)

    # for cut
    ego_center = list(ego_info.vehicles_info[car_id]["center"])[:2]

    if is_deleted_cp_vehicle(ego_info, cp_info, car_id):
        CLogger.info("delete cp vehicle!")
        return False, 0, 0

    cp_car_id, cp_delete_flag = common.find_cp_vehicle_id(cp_info, car_id)

    ego_obj_mesh = delete_obj_from_pcd(ego_info, car_id)

    ego_info.delete_vehicle_of_id(car_id)

    if cp_delete_flag:
        CLogger.info(f"------------- delete cooperative vehicle id = {cp_car_id}")
        if cp_info.param['vehicles'][cp_car_id]['ass_id'] != car_id:
            sys.exit()
        cp_center = list(cp_info.vehicles_info[cp_car_id]["center"])[:2]
        cp_obj_mesh = delete_obj_from_pcd(cp_info, cp_car_id)
        cp_info.delete_vehicle_of_id(cp_car_id)

    if ego_obj_mesh is not None:
        part_result, full_result = occ.delete_occlusion_detect(ego_info, cp_info, ego_obj_mesh)
        insert_after_delete(ego_info, cp_info, part_result, full_result)

    if cp_delete_flag and cp_obj_mesh is not None:
        cp_part_result, cp_full_result = occ.delete_occlusion_detect(cp_info, ego_info, cp_obj_mesh)
        insert_after_delete(cp_info, ego_info, cp_part_result, cp_full_result)

    # visualize after delete
    # cp_corner = common.points_system_transform(corner, ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])
    # vis.show_ego_and_cp_with_corner(cp_info, ego_info, cp_corner)
    # vis.show_obj_with_corner(ego_info, corner)
    # vis.show_obj_with_corner(cp_info, cp_corner)

    if not cp_delete_flag:
        return True, ego_center, [0, 0]

    return True, ego_center, cp_center

# 实现带有“背景修补”的点云删除
def delete_obj_from_pcd(v2x_info, car_id):
    bg_pcd = pc_numpy_2_o3d(v2x_info.pc)
    road_pcd = pc_numpy_2_o3d(v2x_info.road_pc)

    vehicle_pcd = common.get_obj_pcd_from_corner(v2x_info.vehicles_info[car_id]["corner"], bg_pcd)

    try:
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(vehicle_pcd)
        mesh_obj = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(vehicle_pcd, 10, tetra_mesh, pt_map)
    except:
        return None

    road_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(road_pcd, 10)
    pcd_inter = road_mesh.sample_points_uniformly(number_of_points=50000)

    occ_delete_mask, shadow_mesh = get_delete_points_idx(mesh_obj, pcd_inter)

    pcd_inter.points = o3d.utility.Vector3dVector(np.array(pcd_inter.points)[occ_delete_mask])

    cropped_bg = np.asarray(common.pcd_to_np(bg_pcd))[
        ~np.isin(np.asarray(bg_pcd.points), np.asarray(vehicle_pcd.points)).all(axis=1)]
    cropped_bg = pc_numpy_2_o3d(cropped_bg)

    try:
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_inter)
        shadow_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_inter, 5000, tetra_mesh, pt_map)
        # vis.show_mesh_with_pcd(shadow_mesh, cropped_bg)
    except:
        v2x_info.pc = common.pcd_to_np(cropped_bg)
        return shadow_mesh

    try:
        lidar_pcd = lidar_simulation(shadow_mesh)
    except:
        CLogger.info("there are no radar lines in this mesh area, straight return!")
        v2x_info.pc = common.pcd_to_np(cropped_bg)
        return shadow_mesh

    obj_pcd = lidar_intensity_convert(cropped_bg, lidar_pcd)

    v2x_info.pc = common.pcd_to_np(obj_pcd + cropped_bg)

    return shadow_mesh

# 处理“露出来的”车辆补全
def insert_after_delete(deleted_v2x_info, other_v2x_info, part_result, full_result):
    if len(part_result) != 0:
        for car_id in part_result:
            CLogger.info(f"insert for part occlusion, car id = {car_id}")
            print(car_id, deleted_v2x_info.vehicles_info.keys())
            vehicle_info = deleted_v2x_info.vehicles_info[car_id]
            success_flag, _, _, combined_pc = insert_obj(deleted_v2x_info, list(vehicle_info['center'][:2]), False, True,
                                              -vehicle_info['yaw_degree'], vehicle_info['corner'])
            if success_flag and combined_pc is not None:
                deleted_v2x_info.pc = combined_pc

    if len(full_result) != 0:
        for car_id in full_result:
            CLogger.info(f"insert for fully occlusion, car id = {car_id}")
            vehicle_info = other_v2x_info.vehicles_info[car_id]

            position = common.center_system_transform(vehicle_info['center'], other_v2x_info.param['lidar_pose'],
                                                      deleted_v2x_info.param['lidar_pose'])
            rz_degree = common.rz_degree_system_transform(vehicle_info['yaw_degree'], other_v2x_info.param['lidar_pose'],
                                                          deleted_v2x_info.param['lidar_pose'])

            success_flag, _, _, combined_pc = insert_obj(deleted_v2x_info, list(position[:2]), False, True,
                                                         -rz_degree, vehicle_info['corner'])
            if success_flag and combined_pc is not None:
                deleted_v2x_info.pc = combined_pc


def is_deleted_cp_vehicle(ego_info, cp_info, ego_id):
    ego_corner = ego_info.vehicles_info[ego_id]['corner']
    cp_corner = common.points_system_transform(ego_corner, ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])
    if is_box_containing_position(cp_corner, [0, 0, 0]):
        return True
    return False

# 基准baseline删除函数
def base_delete(ego_info, cp_info, car_id):
    """
    baseline delete transformation
    """
    CLogger.info(f"background index = {ego_info.bg_index}, delete ego vehicle id = {car_id}")

    corner = ego_info.vehicles_info[car_id]["corner"]
    # for cut
    ego_center = list(ego_info.vehicles_info[car_id]["center"])[:2]

    # vis.show_ego_and_cp_with_corner(ego_info, cp_info, corner)

    cp_car_id, find_flag = common.find_cp_vehicle_id(cp_info, car_id)

    delete_obj_simple(ego_info, car_id)

    ego_info.delete_vehicle_of_id(car_id)

    if find_flag:
        CLogger.info(f"delete cooperative vehicle id = {cp_car_id}")
        cp_center = list(cp_info.vehicles_info[cp_car_id]["center"])[:2]
        delete_obj_simple(cp_info, cp_car_id)
        cp_info.delete_vehicle_of_id(cp_car_id)
        # vis.show_obj_with_car_id(cp_info, cp_car_id)
        return ego_center, cp_center

    return ego_center, [0, 0]

    # visualize after baseline delete
    # cp_corner = common.points_system_transform(corner, ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])
    # vis.show_ego_and_cp_with_corner(ego_info, cp_info, corner)
    # vis.show_obj_with_corner(ego_info, corner)
    # vis.show_obj_with_corner(cp_info, cp_corner)


def delete_obj_simple(v2x_info, car_id):
    bg_pcd = pc_numpy_2_o3d(v2x_info.pc)

    vehicle_pcd = common.get_obj_pcd_from_corner(v2x_info.vehicles_info[car_id]["corner"], bg_pcd)

    cropped_bg = np.asarray(bg_pcd.points)[
        ~np.isin(np.asarray(bg_pcd.points), np.asarray(vehicle_pcd.points)).all(axis=1)]
    cropped_bg = pc_numpy_2_o3d(cropped_bg)

    v2x_info.pc = np.asarray(cropped_bg.points)     # only use (x, y, z) of pcd


if __name__ == '__main__':
    # delete test
    select_obj_list = []
    select_data_num = config.v2x_config.select_data_num

    index_list = list(range(1, select_data_num + 1))

    for bg_index in range(1, select_data_num + 1):
        index_list.remove(bg_index)
        ego_info = V2XInfo(bg_index)
        cp_info = V2XInfo(bg_index, is_ego=False)
        vehicle_num = len(ego_info.param)

        # random get car index
        car_index = random.randint(0, vehicle_num - 1)

        base_delete(ego_info, cp_info, car_index)
