import config
import random
import numpy as np
import core.obj_delete as delete
import core.obj_insert as insert
import utils.visual as vis
import utils.common_utils as common
from utils.v2x_object import V2XInfo
from logger import CLogger


def vehicle_insert(ego_info, cp_info):# 在道路上随机生成位置并直接插入点云。
    """
    Baseline insert (insert obj on road where baseline setting)

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :return: car id for cut (if needed)
    """
    success_flag = False
    count = 1

    # loop while success   循环直至找到成功插入的位置为止
    while not success_flag:
        CLogger.info(f"try baseline insert {count} times...")
        pos = insert.generate_base_insert_pos(ego_info.pc[:, :3])   # 路上寻找合法位置
        rz_degree = np.random.uniform(-180, 180)     #随机生成偏航角
        success_flag, ego_id, cp_id = insert.base_insert(ego_info, cp_info, pos, rz_degree)  # 在ego和cp的数据中同步插入车辆
        if success_flag:
            return ego_id, cp_id
        count += 1


def vehicle_delete(ego_info, cp_info, car_id=0): # 直接移除车辆点云，不处理扫描线遮挡逻辑。
    """
    Baseline delete, only the vehicle is deleted, and the lidar scan line is not processed.

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :param car_id: delete object of car_id
    :return: operation center for cut (if needed)
    """
    return delete.base_delete(ego_info, cp_info, car_id)

# 直接操作点云数组，将点云坐标乘以缩放因子或旋转矩阵，并同步更新 JSON 标注
def vehicle_translate(ego_info, cp_info, car_id=0, translate=None):
    """
    Baseline translation, translate the vehicle point to the specified location.

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :param car_id: translate object of car_id
    :param translate: the same location as V2XGen translate
    :return: car id for cut (if needed)
    """
    # 找到该车在ego坐标系下的点云索引
    # 对这些点应用【x，y，0】的位移
    if translate is None:
        translate = [0, 0]
    CLogger.info(f"Background index = {ego_info.bg_index}, translate vehicle car id = {car_id}")
    ego_car_id = car_id

    cp_car_id, flag = common.find_cp_vehicle_id(cp_info, ego_car_id)

    translate_vector = [translate[0], translate[1], 0]

    ego_corner = ego_info.vehicles_info[ego_car_id]["corner"]
    if ego_car_id in ego_info.vehicles_info:
        vis_corner = ego_info.vehicles_info[ego_car_id]['corner']
    else:
        # 如果不存在，尝试从 cp_info 获取或设为 None
        vis_corner = None
    ego_obj_idx = common.get_pc_index_in_corner(ego_info.pc, ego_corner)
    #原 ego_info.pc[ego_obj_idx] += translate_vector
    ego_info.pc[ego_obj_idx, :3] += translate_vector[:3]
    for i in range(len(ego_info.param["vehicles"][ego_car_id]["location"])):
        ego_info.param["vehicles"][ego_car_id]["location"][i] += translate_vector[i]    # update center param
    ego_info.load_vehicles_info()   # reload vehicle info

    # translate cooperative object  如果cv也能看到这辆车，同步对其进行平移，双端一致
    if flag:
        cp_corner = cp_info.vehicles_info[cp_car_id]["corner"]
        cp_obj_idx = common.get_pc_index_in_corner(cp_info.pc, cp_corner)
        #原 cp_info.pc[cp_obj_idx] += translate_vector
        cp_info.pc[cp_obj_idx, :3] += translate_vector[:3]
        for i in range(len(ego_info.param["vehicles"][ego_car_id]["location"])):
            cp_info.param["vehicles"][cp_car_id]["location"][i] += translate_vector[i]
        cp_info.load_vehicles_info()

        # visualize after baseline translate
        # vis.show_ego_and_cp_for_translation(ego_info, cp_info, ego_car_id, vis_corner)
        # vis.show_obj_for_translation(ego_info, ego_car_id, vis_corner)
        # vis_cp_corner = common.points_system_transform(vis_corner, ego_info.param['lidar_pose'],
        #                                                cp_info.param['lidar_pose'])
        # vis.show_obj_for_translation(cp_info, cp_car_id, vis_cp_corner)
        return ego_car_id, cp_car_id

    return ego_car_id, -1


def vehicle_scaling(ego_info, cp_info, car_id=0, scaling_ratio=1):
    """
    Baseline scaling，scale the vehicle point in the center of the vehicle box.

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :param car_id: scale object of car_id
    :return: car id for cut (if needed)
    :param scaling_ratio: the same ratio as V2XGen translate
    """
    CLogger.info(f"Background index = {ego_info.bg_index}, baseline scaling vehicle car id = {car_id}")
    ego_corner = ego_info.vehicles_info[car_id]["corner"]
    ego_obj_idx = common.get_pc_index_in_corner(ego_info.pc, ego_corner)
    pts = ego_info.pc[ego_obj_idx, :3]  # 计算点云相对于车辆中心 pts_center 的向量

    cp_car_id, flag = common.find_cp_vehicle_id(cp_info, car_id)

    pts_center = ego_info.vehicles_info[car_id]["center"]
    vectors = pts - pts_center

    # scaling vector   向量乘以缩放系数 scaling_ratio
    scaled_vectors = vectors * scaling_ratio
    ego_info.pc[ego_obj_idx, :3] = scaled_vectors + pts_center
    ego_extent = ego_info.param["vehicles"][car_id]['extent']
    # 更新 JSON 中的 extent（长宽高）参数。
    ego_info.param["vehicles"][car_id]['extent'] = \
        [ego_extent[0] * scaling_ratio, ego_extent[1] * scaling_ratio, ego_extent[2] * scaling_ratio]
    ego_info.load_vehicles_info()  # reload vehicle information

    # cooperative vehicle scaling
    if flag:
        cp_corner = cp_info.vehicles_info[cp_car_id]["corner"]
        cp_obj_idx = common.get_pc_index_in_corner(cp_info.pc, cp_corner)
        pts = cp_info.pc[cp_obj_idx, :3]
        pts_center = cp_info.vehicles_info[cp_car_id]["center"]
        vectors = pts - pts_center
        scaled_vectors = vectors * scaling_ratio
        cp_info.pc[cp_obj_idx, :3] = scaled_vectors + pts_center
        cp_extent = cp_info.param["vehicles"][cp_car_id]['extent']
        cp_info.param["vehicles"][cp_car_id]['extent'] = \
            [cp_extent[0] * scaling_ratio, cp_extent[1] * scaling_ratio, cp_extent[2] * scaling_ratio]
        cp_info.load_vehicles_info()

        # visualize after transformation
        # vis.show_ego_and_cp_with_id(ego_info, cp_info, car_id, cp_car_id)
        # vis.show_obj_with_car_id(ego_info, car_id)
        # vis.show_obj_with_car_id(cp_info, cp_car_id)
        return car_id, cp_car_id

    return car_id, -1


def vehicle_rotation(ego_info, cp_info, car_id, ego_rz_degree, cp_rz_degree):
    """
    Baseline rotation，rotate the vehicle point in the center of the vehicle box.

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :param car_id: scale object of car_id
    :param ego_rz_degree: ego vehicle rotate degree
    :param cp_rz_degree: cooperative vehicle rotate degree
    :return:
    """
    CLogger.info(f"Background index = {ego_info.bg_index}, baseline rotate vehicle car id = {car_id}")
    ego_corner = ego_info.vehicles_info[car_id]["corner"]
    ego_obj_idx = common.get_pc_index_in_corner(ego_info.pc, ego_corner)

    cp_car_id, flag = common.find_cp_vehicle_id(cp_info, car_id)

    # rotation matrix  构建2D旋转矩阵R
    theta = np.radians(ego_rz_degree)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    pts = ego_info.pc[ego_obj_idx, :3]
    pts_center = ego_info.vehicles_info[car_id]["center"]
    vectors = pts - pts_center

    # rotate points in corner 使用矩阵乘法 np.dot(vectors, R.T) 旋转车辆点云
    rotated_vectors = np.dot(vectors, R.T)
    ego_info.pc[ego_obj_idx, :3] = rotated_vectors + pts_center
    ego_info.param["vehicles"][car_id]["angle"][1] = np.degrees(theta)  #更新 JSON 的 angle 偏航角
    ego_info.load_vehicles_info()

    # cooperative rotation
    if flag:
        cp_theta = np.radians(cp_rz_degree)
        R = np.array([
            [np.cos(cp_theta), -np.sin(cp_theta), 0],
            [np.sin(cp_theta), np.cos(cp_theta), 0],
            [0, 0, 1]
        ])
        cp_corner = cp_info.vehicles_info[cp_car_id]["corner"]
        cp_obj_idx = common.get_pc_index_in_corner(cp_info.pc, cp_corner)
        pts = cp_info.pc[cp_obj_idx,:3]
        pts_center = cp_info.vehicles_info[cp_car_id]["center"]
        vectors = pts - pts_center
        rotated_vectors = np.dot(vectors, R.T)
        cp_info.pc[cp_obj_idx, :3] = rotated_vectors + pts_center
        cp_info.param["vehicles"][cp_car_id]["angle"][1] = np.degrees(cp_theta)
        cp_info.load_vehicles_info()

        # visualize after transformation
        # vis.show_ego_and_cp_with_id(ego_info, cp_info, car_id, cp_car_id)
        # vis.show_obj_with_car_id(ego_info, car_id)
        # vis.show_obj_with_car_id(cp_info, cp_car_id)
        return car_id, cp_car_id

    return car_id, -1


if __name__ == '__main__':
    # function test
    select_obj_list = []
    select_data_num = config.v2x_config.select_data_num

    index_list = list(range(1, select_data_num + 1))

    # 循环遍历数据集中的每一帧（bg_index）       为每一帧分别创建 Ego 和 CP 的 V2XInfo 对象
    for bg_index in range(1, select_data_num + 1):
        index_list.remove(bg_index)
        ego_obj = V2XInfo(bg_index)
        cp_obj = V2XInfo(bg_index, is_ego=False)
        vehicle_num = len(ego_obj.param)

        # random get car index 随机选择一辆车
        car_index = random.randint(0, vehicle_num - 1)

        vehicle_translate(ego_obj, cp_obj, car_index)
