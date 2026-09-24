import numpy as np
import open3d as o3d
import config
import utils.common_utils as common
import opencood.utils.common_utils as common_utils
import torch

# 作用：同时显示一个 3D 网格（Mesh）和一个点云。
# 关键点：会自动计算网格的最小包围盒（OBB）并显示，常用于检查 3D 模型与原始采样点云的拟合程度。
def show_mesh_with_pcd(mesh, pcd):
    """
    Visualize a 3D mesh and a point cloud in an interactive window.

    :param mesh: Open3D TriangleMesh object to visualize
    :param pcd: Open3D PointCloud object to visualize alongside the mesh
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    vis.add_geometry(pcd)

    box3d = mesh.get_minimal_oriented_bounding_box()
    mesh.compute_vertex_normals()

    vis.add_geometry(mesh)
    vis.add_geometry(box3d)

    vis.run()
    vis.destroy_window()

# 作用：显示网格及其最小外接矩形框。
# 关键点：会在控制台打印出包围盒的 8 个顶点坐标和高度（Height），用于精确核对几何尺寸。
def show_mesh_with_box(mesh_obj):
    """
    Visualize a 3D mesh and its minimal oriented bounding box.

    :param mesh_obj: Open3D TriangleMesh object to visualize with its bounding box
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    box3d = mesh_obj.get_minimal_oriented_bounding_box()
    box_points = box3d.get_box_points()

    # box_mesh.compute_vertex_normals()
    points = np.asarray(box_points)
    print(points)
    print("height = ", np.ptp(points[:, 2]))

    mesh_obj.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_obj])
    vis.add_geometry(mesh_obj)

    vis.add_geometry(box3d)

    # vis.add_geometry(mixed_pcd)
    vis.run()
    vis.destroy_window()

# 作用：通用的点云与 Box 显示函数。
# 关键点：将 Numpy 格式的点云转换为 Open3D 格式，并将点云渲染为橙色（V2X 场景中常用的主车颜色）。
def show_pc_with_box(pc, box):
    """
    Visualize a numpy point cloud and a 3D bounding box.

    :param pc: Numpy array of shape (N, 3) or (N, 4) representing the point cloud
    :param box: Open3D geometry (e.g., AxisAlignedBoundingBox, OrientedBoundingBox) for visualization
    """
    pcd = common.pc_numpy_2_o3d(pc)
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)
    vis.add_geometry(box)
    rgb_color = [245 / 255, 144 / 255, 1 / 255]

    pcd.paint_uniform_color(rgb_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# 作用：可视化当前帧的完整背景点云和所有车辆的边界框。
# 场景：用于检查标注信息是否与点云中的物体位置对齐。
def show_bg_with_boxes(v2x_info):
    """
    Visualize the background point cloud and all vehicle bounding boxes from a V2XInfo object.

    :param v2x_info: V2XInfo object containing background point cloud and vehicle info
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    for val in v2x_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    pcd = common.pc_numpy_2_o3d(v2x_info.pc)

    rgb_color = [245/255, 144/255, 1/255]

    pcd.paint_uniform_color(rgb_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# 作用：在点云中突出显示特定 ID 的车辆。
# 场景：当需要追踪或调试某一特定车辆的属性时使用。
def show_obj_with_car_id(v2x_info, car_id):
    """
    Visualize the background point cloud and the bounding box of a specific vehicle.

    :param v2x_info: V2XInfo object containing point cloud and vehicle info
    :param car_id: ID of the specific vehicle to highlight
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    corner = v2x_info.vehicles_info[car_id]['corner']
    line_set = common.corner_to_line_set_box(corner)
    vis.add_geometry(line_set)

    # pcd.paint_uniform_color([0, 0, 0])
    pcd = common.pc_numpy_2_o3d(v2x_info.pc)
    if v2x_info.is_ego:
        pcd_color = [245 / 255, 144 / 255, 1 / 255]
    else:
        pcd_color = [1, 1, 1]
    pcd.paint_uniform_color(pcd_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# 作用：自定义 Box 显示。它不使用线框，而是用红色的圆柱体（Cylinders）拼接成一个 Box 边框。
# 场景：这种显示方式视觉上更粗、更明显，适合在复杂背景下观察 Box 位置。
def show_obj_with_corner(v2x_info, corner):
    """
    Visualize the background point cloud and a custom bounding box (as cylinders).

    :param v2x_info: V2XInfo object containing the background point cloud
    :param corner: List/array of 8 (x,y,z) coordinates defining the bounding box corners
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size

    # render.background_color = np.array(config.lidar_config.render_background_color)

    # line_set = common.corner_to_line_set_box(corner)
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7],
                          [4, 5], [5, 6], [6, 7], [7, 4]])

    cylinders = []

    for line in lines_box:
        point1 = corner[line[0]]
        point2 = corner[line[1]]

        cylinder = common.create_cylinder_between_points(point1, point2, radius=0.03)

        cylinder.paint_uniform_color([1, 0, 0])
        cylinders.append(cylinder)

    mesh = o3d.geometry.TriangleMesh()
    for cyl in cylinders:
        mesh += cyl
    # vis.add_geometry(line_set)
    vis.add_geometry(mesh)

    # pcd.paint_uniform_color([0, 0, 0])
    pcd = common.pc_numpy_2_o3d(v2x_info.pc)
    if v2x_info.is_ego:
        pcd_color = [0, 0, 1]
        # pcd_color = [245 / 255, 144 / 255, 1 / 255]
    else:
        pcd_color = [0, 100 / 255, 0]
    pcd.paint_uniform_color(pcd_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# 作用：与 show_bg_with_boxes 类似，是点云和所有车辆 Box 的标准可视化。
def show_pc(v2x_info):
    """
    Visualize the background point cloud and all vehicle bounding boxes (simplified).

    :param v2x_info: V2XInfo object containing background point cloud and vehicle info
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    for val in v2x_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    ego_pcd = common.pc_numpy_2_o3d(v2x_info.pc)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    vis.add_geometry(ego_pcd)

    vis.run()
    vis.destroy_window()

# 作用：点云融合显示。
# 核心逻辑：通过外参矩阵计算变换矩阵 T(cp2ego) = P逆(ego) · P(cp) ,将协作端的点云变换到主车坐标系下。
# 颜色区分：主车点云为橙色，协作端点云为白色。用于检查两端数据在空间上是否重合对齐。
def show_ego_and_cp_pc(ego_info, cp_info):
    """
    Visualize point clouds and vehicle boxes from both ego and cooperative vehicles.

    :param ego_info: V2XInfo object for the ego vehicle
    :param cp_info: V2XInfo object for the cooperative vehicle
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    for val in ego_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    for val in cp_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    # pcd.paint_uniform_color([0, 0, 0])
    T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    vis.add_geometry(ego_pcd)
    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()

# 作用：在两端点云融合的基础上，显示特定的车辆 ID 或自定义的 Corner 框。
# 场景：验证协作端共享的车辆目标在主车视角下的准确性。
def show_ego_and_cp_with_id(ego_info, cp_info, ego_id, cp_id):
    """
    Visualize ego/coop point clouds and highlight specific vehicle IDs.

    :param ego_info: V2XInfo object for the ego vehicle
    :param cp_info: V2XInfo object for the cooperative vehicle
    :param ego_id: ID of the specific vehicle to highlight in the ego object
    :param cp_id: ID of the specific vehicle to highlight in the coop object (unused in current code)
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    # pcd.paint_uniform_color([0, 0, 0])
    T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    corner = ego_info.vehicles_info[ego_id]['corner']
    line_set = common.corner_to_line_set_box(corner)
    vis.add_geometry(line_set)

    # corner = cp_info.vehicles_info[cp_id]['corner']
    # line_set = common.corner_to_line_set_box(corner)
    # vis.add_geometry(line_set)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


def show_ego_and_cp_with_corner(ego_info, cp_info, corner):
    """
    Visualize ego and cooperative point clouds with a custom bounding box.

    :param ego_info: V2XInfo object containing ego vehicle's point cloud and parameters
    :param cp_info: V2XInfo object containing cooperative vehicle's point cloud and parameters
    :param corner: List/array of 8 (x,y,z) coordinates defining the custom bounding box corners
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    # render.background_color = np.array(config.lidar_config.render_background_color)
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7],
                          [4, 5], [5, 6], [6, 7], [7, 4]])

    cylinders = []

    for line in lines_box:
        point1 = corner[line[0]]
        point2 = corner[line[1]]

        cylinder = common.create_cylinder_between_points(point1, point2, radius=0.05)

        cylinder.paint_uniform_color([1, 0, 0])
        cylinders.append(cylinder)

    mesh = o3d.geometry.TriangleMesh()
    for cyl in cylinders:
        mesh += cyl
    # vis.add_geometry(line_set)
    vis.add_geometry(mesh)

    # pcd.paint_uniform_color([0, 0, 0])
    T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    # ego_color = [0, 0, 1]
    cp_color = [0, 0, 1]
    ego_color = [0, 75 / 255, 0]
    ego_pcd.paint_uniform_color(ego_color)
    # cp_color = [0, 100 / 255, 0]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    line_set = common.corner_to_line_set_box(corner)
    vis.add_geometry(line_set)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()

# 作用：在融合的点云场景中，同时查看原始位置和变换后的位置。
def show_ego_and_cp_for_translation(ego_info, cp_info, car_id, corner):
    """
    Visualize ego and cooperative point clouds with vehicle bounding boxes for translation check.

    :param ego_info: V2XInfo object for the ego vehicle
    :param cp_info: V2XInfo object for the cooperative vehicle
    :param car_id: ID of the vehicle in ego_info to visualize
    :param corner: Coordinates of the translated bounding box corners to visualize
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    ego_corner = ego_info.vehicles_info[car_id]['corner']
    ego_line_set = common.corner_to_line_set_box(ego_corner)
    vis.add_geometry(ego_line_set)

    line_set = common.corner_to_line_set_box(corner, [1, 1, 1])
    vis.add_geometry(line_set)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


# 作用：对比显示原始 Box 和 平移后的 Box。
# 场景：在执行 vehicle_translation 操作后，通过此函数肉眼确认平移的方向和距离是否符合预期。
def show_obj_for_translation(v2x_info, car_id, corner):
    """
    Visualize a vehicle's original and translated bounding boxes.

    :param v2x_info: V2XInfo object containing the point cloud and vehicle information
    :param car_id: ID of the vehicle to visualize
    :param corner: Coordinates of the translated bounding box corners
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    pcd = common.pc_numpy_2_o3d(v2x_info.pc)

    if v2x_info.is_ego:
        pcd_color = [245 / 255, 144 / 255, 1 / 255]
    else:
        pcd_color = [1, 1, 1]
    pcd.paint_uniform_color(pcd_color)

    cur_corner = v2x_info.vehicles_info[car_id]['corner']
    cur_line_set = common.corner_to_line_set_box(cur_corner)
    vis.add_geometry(cur_line_set)

    line_set = common.corner_to_line_set_box(corner, [1, 1, 1])
    vis.add_geometry(line_set)

    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

def show_ego_and_cp_for_translation_new(ego_info, cp_info, car_id, original_corner, new_corner):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    T_cp2ego = np.linalg.inv(ego_info.param["lidar_pose"]) @ cp_info.param["lidar_pose"]
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    # 渲染原始框 (红色)
    orig_line_set = common.corner_to_line_set_box(original_corner) # 默认红
    vis.add_geometry(orig_line_set)
    
    # 渲染目标框 (白色)
    new_line_set = common.corner_to_line_set_box(new_corner, [1, 1, 1])
    vis.add_geometry(new_line_set)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


def show_obj_for_translation_new(v2x_info, old_corner, new_corner):
    """
    在单端点云视角下，对比显示车辆的原始位置和变换后的位置。

    :param v2x_info: V2XInfo 对象，包含点云数据
    :param old_corner: 备份的原始 Box 坐标 (8, 3) -> 显示为红色
    :param new_corner: 变换后的目标 Box 坐标 (8, 3) -> 显示为白色
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()
    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    # 1. 渲染背景点云
    pcd = common.pc_numpy_2_o3d(v2x_info.pc)
    if v2x_info.is_ego:
        pcd_color = [245 / 255, 144 / 255, 1 / 255] # 橙色
    else:
        pcd_color = [1, 1, 1] # 白色
    pcd.paint_uniform_color(pcd_color)
    vis.add_geometry(pcd)

    # 2. 渲染原始位置框 (强制红色 [1, 0, 0])
    # 不再使用 v2x_info.vehicles_info[car_id]，直接用传入的 old_corner
    old_line_set = common.corner_to_line_set_box(old_corner, [1, 0, 0])
    vis.add_geometry(old_line_set)

    # 3. 渲染目标位置框 (强制白色 [1, 1, 1])
    # 直接使用传入的 new_corner
    new_line_set = common.corner_to_line_set_box(new_corner, [1, 1, 1])
    vis.add_geometry(new_line_set)

    vis.run()
    vis.destroy_window()

def visualize_and_eval_post_operation(v2x_info, pred_box_tensor, pred_scores, gt_box_tensor, 
                                      iou_threshold=0.5, is_ego=True, window_name="Post-Operation Evaluation"):
    """
    在insert/translation/scaling/rotation操作后，可视化点云、预测框和GT框的重合程度，
    并统计基于IoU阈值的正确预测数量（TP）。

    :param v2x_info: V2XInfo对象，包含点云 (pc) 和车辆信息 (vehicles_info)
    :param pred_box_tensor: 模型预测的边界框张量 (N, 8, 3) torch.Tensor 或 np.ndarray
    :param pred_scores: 预测框置信度 (N,)
    :param gt_box_tensor: 操作后的GT边界框张量 (M, 8, 3) torch.Tensor 或 np.ndarray
    :param iou_threshold: IoU阈值，用于过滤和统计
    :param is_ego: 是否为ego车辆（影响点云颜色）
    :param window_name: 窗口名称
    """
    # 1. 数据是否是torch.Tensor，是的话转换为Numpy
    if isinstance(pred_box_tensor, torch.Tensor):
        pred_boxes = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    else:
        pred_boxes = pred_box_tensor

    if isinstance(pred_scores,torch.Tensor):
        pred_scores = common_utils.torch_tensor_to_numpy(pred_scores)
    else:
        pred_scores = pred_scores
    
    if isinstance(gt_box_tensor, torch.Tensor):
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_box_tensor)
    else:
        gt_boxes = gt_box_tensor

    if len(pred_boxes) == 0:
        print("No predictions detected.")
        
    # 2. 计算IoU并筛选有效预测框（TP统计）
    det_polygons = common_utils.convert_format(pred_boxes)  # 转换为Shapely Polygon (BEV)
    gt_polygons = common_utils.convert_format(gt_boxes)
    
    valid_pred_indices = []  # IoU >= 阈值的预测框索引
    tp_count = 0  # 正确预测数量（TP）
    matched_gt_indices = set()  # 避免一个GT匹配多个预测
    for i, det_poly in enumerate(det_polygons):
        if len(gt_polygons) == 0:
            break
        ious = common_utils.compute_iou(det_poly, gt_polygons)
        max_iou = np.max(ious)   # 寻找最佳匹配
        if max_iou >= iou_threshold:  # 最大 IoU 超过设定的阈值（例如 0.5）时，才认为这个预测可能是一个 TP
            gt_idx = np.argmax(ious)
            if gt_idx not in matched_gt_indices:  # 1:1匹配 检查这个 gt_idx 是否已经被之前的预测框“占坑”了
                valid_pred_indices.append(i)
                matched_gt_indices.add(gt_idx) # 如果没有被占用，记录该预测框索引 i，并将该真值标记为已匹配
                tp_count += 1

    # 打印统计结果
    print(f"Post-Operation Evaluation (IoU Threshold: {iou_threshold}):")
    print(f"Total GT Boxes: {len(gt_boxes)}")
    print(f"Total Pred Boxes: {len(pred_boxes)}")
    print(f"Correct Predictions (TP): {tp_count} (IoU >= {iou_threshold})")

    # 3. 创建Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)
    
    render = vis.get_render_option()
    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    if isinstance(v2x_info, dict):
        pc_raw = v2x_info.get('pc', v2x_info.get('origin_lidar'))
    else:
        pc_raw = v2x_info.pc
    
    # 统一转为 CPU Numpy
    if torch.is_tensor(pc_raw):
        pc_np = pc_raw.detach().cpu().numpy()
    else:
        pc_np = np.array(pc_raw)

    if pc_np.ndim == 3:
        pc_np = pc_np.squeeze(0)
    # pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])

    pc_for_o3d = np.ascontiguousarray(pc_np[:, :3], dtype=np.float64)
    # 渲染背景点云
    pcd = o3d.geometry.PointCloud()
    try:
        pcd.points = o3d.utility.Vector3dVector(pc_for_o3d)
    except RuntimeError as e:
        print(f"Open3D casting error: {e}")
        pcd.points = o3d.utility.Vector3dVector(pc_for_o3d.astype(np.float64))
    pcd_color = [245 / 255, 144 / 255, 1 / 255] if is_ego else [1, 1, 1]  # ego橙色，其他白色
    pcd.paint_uniform_color(pcd_color)
    vis.add_geometry(pcd)

    # 定义边界框线序
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], 
             [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    # 渲染GT框（红色）
    for box in gt_boxes:
        ls = o3d.geometry.LineSet()
        box_np = np.ascontiguousarray(box, dtype=np.float64)
        ls.points = o3d.utility.Vector3dVector(box)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color([1, 0, 0])  # 红色
        vis.add_geometry(ls)

    # 渲染有效预测框（绿色，只渲染IoU >= 阈值的）
    for idx in valid_pred_indices:
        ls = o3d.geometry.LineSet()
        pred_box_np = np.ascontiguousarray(pred_boxes[idx], dtype=np.float64)
        ls.points = o3d.utility.Vector3dVector(pred_boxes[idx])
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color([0, 1, 0])  # 绿色
        vis.add_geometry(ls)

    # 弹出窗口并运行
    vis.run()
    vis.destroy_window()

def visualize_individual_perspectives(batch_data, ego_preds, cp_preds, frame_idx):
    """
    先后展示主车视角和协作车视角的感知结果。
    
    :param batch_data: 完整的 batch 数据字典
    :param ego_preds: tuple (pred_box, pred_score, gt_box) 为主车结果
    :param cp_preds: tuple (pred_box, pred_score, gt_box) 为协作车结果
    :param frame_idx: 当前帧序号
    """
    
    # 处理单个视角的渲染
    def render_single_view(v2x_dict, preds, window_name, is_ego=True):
        if preds[0] is None or len(preds[0]) == 0:
            print(f"Warning: No predictions to show for {window_name}")
            return

        # 1. 提取并转换点云 (关键修复：移动到 CPU)
        # pc_raw = v2x_dict.get('pc', v2x_dict.get('lidar_np'))
        pc_raw = None
    # 按照优先级尝试获取原始点云
        search_keys = ['pc', 'lidar_np', 'origin_lidar', 'processed_lidar']
    
        for key in search_keys:
            if key in v2x_dict:
                pc_raw = v2x_dict[key]
                # print(f"DEBUG: Key found = {key}")
                # print(f"DEBUG: pc_raw type = {type(pc_raw)}")
                # print(f"DEBUG: pc_raw device = {pc_raw.device if hasattr(pc_raw, 'device') else 'No Device'}")
                break
            
        if pc_raw is None:
            print(f"Error: Could not find point cloud in {v2x_dict.keys()}")
            return

        if isinstance(pc_raw, list):
        # 逐个将列表里的元素搬到 CPU
            processed_list = []
            for item in pc_raw:
                if hasattr(item, 'cpu'):
                    processed_list.append(item.detach().cpu().numpy())
                else:
                    processed_list.append(item)
        
        # 拼接成一个大的 Numpy 矩阵
            if len(processed_list) > 0:
                pc_np = np.concatenate(processed_list, axis=0) if isinstance(processed_list[0], np.ndarray) else np.array(processed_list)
            else:
                pc_np = np.empty((0, 3))

    # 情况 B：pc_raw 直接就是 Tensor
        elif hasattr(pc_raw, 'cpu'):
            pc_np = pc_raw.detach().cpu().numpy()
    
    # 情况 C：已经是 Numpy 或其他
        else:
            pc_np = np.array(pc_raw)

    # --- 后续处理保持不变 ---
    # 再次确保维度正确 (N, 3)
        if pc_np.ndim == 3:
            pc_np = pc_np.squeeze(0)
    
    # 如果列表转换后依然形状不对，强制转为 2D
        if pc_np.ndim == 1 and pc_np.size > 0:
            pc_np = pc_np.reshape(-1, 3)
        
    
    # 检查是否依然是标量（即报错的位置）
        if pc_np.ndim < 2:
            print(f"Error: Point cloud array dimension is too low: {pc_np.ndim}. Value: {pc_np}")
            return

        # 2. 提取预测框和真值框 (确保是 Numpy)
        pred_boxes = preds[0].detach().cpu().numpy() if torch.is_tensor(preds[0]) else preds[0]
        gt_boxes = preds[2].detach().cpu().numpy() if torch.is_tensor(preds[2]) else preds[2]

        # 3. 创建 Open3D 窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)
        
        # 背景和点云设置
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1]) # 深灰色背景
        opt.point_size = 1.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])
        # 主车橙色，协作车白色
        pcd.paint_uniform_color([1, 0.7, 0.2] if is_ego else [0.8, 0.8, 0.8])
        vis.add_geometry(pcd)

        # 4. 绘制框的线序
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], 
                 [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

        # 渲染 GT (红色)
        if gt_boxes is not None and len(gt_boxes) > 0:
            for box in gt_boxes:
            # --- 新增转换逻辑 ---
                #确保 box 是有效的数组/张量且形状正确 (N, 3)
                if not hasattr(box, '__len__') or isinstance(box, (int, float)):
                    continue
                if hasattr(box, 'cpu'):
                    box = box.detach().cpu().numpy()

                # 确保 box 至少是二维的 (8, 3)
                box_np = np.array(box)
                if box_np.ndim != 2:
                    continue
            # ------------------
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(box.astype(np.float64))
                ls.lines = o3d.utility.Vector2iVector(lines)
                ls.paint_uniform_color([1, 0, 0])
                vis.add_geometry(ls)

        # 渲染 Pred (绿色)
        if pred_boxes is not None:
            for box in pred_boxes:
            # --- 新增转换逻辑 ---
                if not hasattr(box, '__len__') or isinstance(box, (int, float)):
                    continue
                if hasattr(box, 'cpu'):
                    box = box.detach().cpu().numpy()

                box_np = np.array(box)
                if box_np.ndim != 2:
                    continue
            # ------------------
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(box.astype(np.float64))
                ls.lines = o3d.utility.Vector2iVector(lines)
                ls.paint_uniform_color([0, 1, 0])
                vis.add_geometry(ls)


        vis.run()
        vis.destroy_window()

    # --- 开始执行展示 ---
    
    # 视角一：主车
    if 'ego' in batch_data:
        render_single_view(batch_data['ego'], ego_preds, f"Frame {frame_idx} - EGO View", is_ego=True)
    
    # 视角二：协作车 (ID: 1)
    if '1' in batch_data:
        render_single_view(batch_data['1'], cp_preds, f"Frame {frame_idx} - CP ID:1 View", is_ego=False)



def calculate_max_iou_for_targets(pred_box_tensor, target_gt_box_tensor):
    """
    在insert/translation/scaling/rotation操作后，可视化点云、预测框和GT框的重合程度，
    并统计基于IoU阈值的正确预测数量（TP）。

    :param v2x_info: V2XInfo对象，包含点云 (pc) 和车辆信息 (vehicles_info)
    :param pred_box_tensor: 模型预测的边界框张量 (N, 8, 3) torch.Tensor 或 np.ndarray
    :param pred_scores: 预测框置信度 (N,)
    :param gt_box_tensor: 操作后的GT边界框张量 (M, 8, 3) torch.Tensor 或 np.ndarray
    """
    # 1. 数据是否是torch.Tensor，是的话转换为Numpy
    if pred_box_tensor is None:
        pred_boxes = np.array([])
    elif isinstance(pred_box_tensor, torch.Tensor):
        pred_boxes = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    else:
        pred_boxes = pred_box_tensor

    if target_gt_box_tensor is None:
        target_gt_box = np.array([])
    elif isinstance(target_gt_box_tensor, torch.Tensor):
        target_gt_box = common_utils.torch_tensor_to_numpy(target_gt_box_tensor)
    else:
        target_gt_box = target_gt_box_tensor

    if len(pred_boxes) == 0 or len(target_gt_box) == 0:
        return np.array([], dtype=np.float32)
        
    # 2. 计算IoU并筛选有效预测框（TP统计）
    det_polygons = common_utils.convert_format(pred_boxes)  # 转换为Shapely Polygon (BEV)
    target_gt_polygons = common_utils.convert_format(target_gt_box)
    
    max_ious = []
    for gt_poly in target_gt_polygons:
        # 计算当前 GT 框与所有预测框的 IoU
        # compute_iou(box, boxes_list) -> 返回 array
        ious = common_utils.compute_iou(gt_poly, det_polygons)
        
        if len(ious) > 0:
            max_iou = np.max(ious)
        else:
            max_iou = 0.0
            
        max_ious.append(max_iou)

    result_array = np.array(max_ious, dtype=np.float32)
    if result_array.ndim == 0:
        result_array = result_array.reshape(-1)
    return result_array

def filter_boxes_by_ids(all_boxes, all_ids, target_ids):
    """
    根据 ID 列表筛选 Box。
    all_boxes: (N, ...)
    all_ids: List[int]
    target_ids: List[int]
    """
    # 如果 all_boxes 为 None，返回空数组
    if all_boxes is None:
        return np.array([])
        
    if len(target_ids) == 0:
        # 返回空数组，保持维度一致 (0, ...)
        if isinstance(all_boxes, np.ndarray):
            return all_boxes[:0] 
        elif isinstance(all_boxes, torch.Tensor):
            return all_boxes[0:0] # 兼容 torch
        else:
            return np.array([])

    # 构建映射
    id_to_idx = {val: idx for idx, val in enumerate(all_ids)}
    
    indices = []
    for tid in target_ids:
        if tid in id_to_idx:
            indices.append(id_to_idx[tid])
        else:
            # 可选：警告 ID 不存在
            pass
            
    if len(indices) == 0:
        if isinstance(all_boxes, np.ndarray):
            return all_boxes[:0]
        elif isinstance(all_boxes, torch.Tensor):
            return all_boxes[0:0]
        else:
            return np.array([])

    # 使用索引切片
    if isinstance(all_boxes, np.ndarray):
        return all_boxes[indices]
    elif isinstance(all_boxes, torch.Tensor):
        return all_boxes[indices]
    else:
        # 如果是 list of lists
        return [all_boxes[i] for i in indices]

