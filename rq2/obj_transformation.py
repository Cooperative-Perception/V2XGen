import numpy as np
import random
import math
import core.obj_delete as delete
import core.obj_insert as insert
import utils.random_param as rand
from core.occlusion_treatment import get_occ_rate_of_obj
from utils.common_utils import center_system_transform, rz_degree_system_transform
from logger import CLogger

# Combine of insert and delete operations to these interfaces.
# RQ2 data generation transformation, for each selected data
# transformation multiple times.


def vehicle_insert(ego_info, cp_info):
    """
    Insert a car at a randomly generated location in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :return: transformation result, True or False
    """
    success_flag = False
    count = 1

    while not success_flag:
        # over 10 times
        if count >= 10:
            return False

        CLogger.info(f"try insert {count} times...")

        position, rz_degree = rand.get_insert_location(ego_info)

        # get id to cut
        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, True, transformation="rotation")
        count += 1

    return True


def vehicle_delete(ego_info, cp_info, car_id=0):
    """
    Delete a car of the car_id in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param car_id: the car id for delete
    :return: transformation result, True or False
    """
    # CLogger.info(f"background index = {ego_info.bg_index}, delete ego vehicle id = {car_id}")
    success_flag, v2x_ego_center, v2x_cp_center = \
        delete.vehicle_delete(ego_info, cp_info, car_id)
    if success_flag:
        return True

    return False


def vehicle_translation(ego_info, cp_info, car_id):
    """
    Translate a car of the car_id to a randomly generated location in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param car_id: the car id for translation
    :return: transformation result, True or False
    """
    success_flag = False
    cnt = 1

    # delete the car in the original location
    while not success_flag:
        if cnt >= 10:
            return False
        success_flag = delete.vehicle_delete(ego_info, cp_info, car_id)
        cnt += 1

    # delete successful
    success_flag = False
    cnt = 1

    while not success_flag:
        # over 10 times
        if cnt >= 10:
            return False

        CLogger.info(f"try translation {cnt} times...")
        # insert car to new location
        position, rz_degree = rand.get_insert_location(ego_info)
        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, True, True, rz_degree, transformation="rotation")

        cnt += 1
    return True


def vehicle_scaling(ego_info, cp_info, car_id):
    """
    Scaling a car of the car_id to a randomly generated scale in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param car_id: the car id for scaling
    :return: transformation result, True or False
    """
    scaling_flag = False
    cnt = 1
    ego_corner = ego_info.vehicles_info[car_id]['corner']
    rz_degree = -ego_info.vehicles_info[car_id]['yaw_degree']
    position = list(ego_info.vehicles_info[car_id]['center'][:2])

    # the range of scaling
    min_ratio, max_ratio = 0.9, 1.1

    # ego_center = ego_info.vehicles_info[car_id]['center']

    # delete the car in the original location
    while not scaling_flag:
        if cnt >= 10:
            return False
        scaling_flag = delete.vehicle_delete(ego_info, cp_info, car_id)
        cnt += 1

    # delete successful
    scaling_flag = False
    cnt = 1

    while not scaling_flag:
        # over 10 times
        if cnt >= 10:
            return False

        ratio = random.uniform(min_ratio, max_ratio)
        CLogger.info(f"try scaling {cnt} times..., scaling ratio = {ratio}")
        corner_center = np.mean(ego_corner, axis=0)
        vectors = ego_corner - corner_center
        scaled_vector = vectors * ratio
        scaling_box = scaled_vector + corner_center

        scaling_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, False, True, rz_degree, scaling_box, transformation="rotation")
        cnt += 1
    return True


def vehicle_rotation(ego_info, cp_info, car_id):
    """
    Rotation a car of the car_id to a randomly generated yaw degree in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param car_id: the car id for translation
    :return: transformation result, True or False
    """
    success_flag = False
    cnt = 1
    org_degree = -ego_info.vehicles_info[car_id]['yaw_degree']
    position = list(ego_info.vehicles_info[car_id]['center'][:2])
    corner = ego_info.vehicles_info[car_id]['corner']

    # delete the car in the original location
    while not success_flag:
        if cnt >= 10:
            return False
        success_flag = delete.vehicle_delete(ego_info, cp_info, car_id)
        cnt += 1

    # delete successful
    success_flag = False
    cnt = 1

    while not success_flag:
        # over 10 times
        if cnt >= 10:
            return False

        rot_degree = rand.get_random_rotation()
        CLogger.info(f"try rotation {cnt} times..., rot degree = {rot_degree}")
        rz_degree = org_degree + rot_degree

        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, False, True, rz_degree, corner, transformation="rotation")

        cnt += 1

    return True


def label_complete_for_ego(ego_info, cp_info):
    """
    Scan each car in the ego tag, calculate the occlusion rate and distance,
    and save it in the label file.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :return:
    """
    # 标签补全（主车视角）。计算每个目标相对于主车的
    # 遮挡率（Occlusion Rate）和距离，并处理坐标系转换

    for car_id, car_info in ego_info.param["vehicles"].items():
        ego_car_center = car_info["location"]
        ego2cp_center = center_system_transform(ego_car_center, ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])
        ego_car_degree = -car_info['angle'][1]

        ego_center_xy = ego_car_center[:2]
        cp_center_xy = ego2cp_center[:2]

        # save occlusion rate and distance
        occ_rate = get_occ_rate_of_obj(ego_info, ego_car_center, ego_car_degree, car_id)
        ego_info.param["vehicles"][car_id]["ego_occ_rate"] = occ_rate

        # ego_distance, cp_distance for ego car
        # 1. delete distance key
        if "distance" in ego_info.param["vehicles"][car_id]:
            ego_info.param["vehicles"][car_id].pop("distance")

        # 2. add ego_distance and cp_distance
        ego_info.param["vehicles"][car_id]["ego_distance"] = math.sqrt(ego_center_xy[0]**2 + ego_center_xy[1]**2)
        ego_info.param["vehicles"][car_id]["cp_distance"] = math.sqrt(cp_center_xy[0]**2 + cp_center_xy[1]**2)

        if occ_rate != 0:
            print("ego = ", ego_info.bg_index, occ_rate)

    # get occ rate for cp cars
    for car_id, car_info in cp_info.param["vehicles"].items():
        if car_info['ass_id'] != -1:
            cp_info.param["vehicles"][car_id]["ego_occ_rate"] = 0
            continue

        # cp car transform to ego system
        cp_car_center = car_info["location"]
        cp2ego_center = center_system_transform(cp_car_center, cp_info.param['lidar_pose'],
                                                ego_info.param['lidar_pose'])
        cp_car_degree = -car_info['angle'][1]
        cp2ego_degree = rz_degree_system_transform(cp_car_degree, cp_info.param['lidar_pose'],
                                                   ego_info.param['lidar_pose'])

        occ_rate = get_occ_rate_of_obj(ego_info, cp2ego_center, cp2ego_degree, car_id)
        cp_info.param["vehicles"][car_id]["ego_occ_rate"] = occ_rate
        if occ_rate != 0:
            print("cp = ", cp_info.bg_index, occ_rate)


def label_complete_for_cp(ego_info, cp_info):
    """
    Scan each car in the cooperative tag, calculate the occlusion rate and distance,
    and save it in the label file.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :return:
    """
    # 标签补全（协作端视角）。计算目标相对于协作端的遮挡率和距离，确保双端标签一致性

    cp_ass_id_list = []

    for car_id, car_info in cp_info.param["vehicles"].items():
        if car_info['ass_id'] != -1:
            cp_ass_id_list.append(car_info['ass_id'])
        cp_car_center = car_info["location"]
        cp2ego_center = center_system_transform(cp_car_center, cp_info.param['lidar_pose'], ego_info.param['lidar_pose'])
        cp_car_degree = -car_info['angle'][1]

        cp_center_xy = cp_car_center[:2]
        ego_center_xy = cp2ego_center[:2]

        # save occlusion rate and distance
        occ_rate = get_occ_rate_of_obj(cp_info, cp_car_center, cp_car_degree, car_id)
        cp_info.param["vehicles"][car_id]["cp_occ_rate"] = occ_rate

        # add ego_distance and cp_distance
        cp_info.param["vehicles"][car_id]["ego_distance"] = math.sqrt(ego_center_xy[0]**2 + ego_center_xy[1]**2)
        cp_info.param["vehicles"][car_id]["cp_distance"] = math.sqrt(cp_center_xy[0]**2 + cp_center_xy[1]**2)

    # get occ rate for cp cars
    for car_id, car_info in ego_info.param["vehicles"].items():
        if car_id in cp_ass_id_list:
            ego_info.param["vehicles"][car_id]["cp_occ_rate"] = 0
            continue

        # ego car transform to cp system
        ego_car_center = car_info["location"]
        ego2cp_center = center_system_transform(ego_car_center,
                                                ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])
        ego_car_degree = -car_info['angle'][1]
        ego2cp_degree = rz_degree_system_transform(ego_car_degree,
                                                   ego_info.param['lidar_pose'], cp_info.param['lidar_pose'])

        occ_rate = get_occ_rate_of_obj(cp_info, ego2cp_center, ego2cp_degree, car_id)
        ego_info.param["vehicles"][car_id]["cp_occ_rate"] = occ_rate




