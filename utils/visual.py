import numpy as np
import open3d as o3d
import config
import utils.common_utils as common


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

