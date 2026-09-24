"""
Template for AnchorGenerator
"""

import numpy as np
import torch

from opencood.utils import box_utils


class BasePostprocessor(object):
    """
    Template for Anchor generator.

    Parameters
    ----------
    anchor_params : dict
        The dictionary containing all anchor-related parameters.
    train : bool
        Indicate train or test mode.

    Attributes
    ----------
    bbx_dict : dictionary
        Contain all objects information across the cav, key: id, value: bbx
        coordinates (1, 7)
    """

    def __init__(self, anchor_params, train=True):
        self.params = anchor_params
        self.bbx_dict = {}
        self.train = train

    def generate_anchor_box(self):
        # needs to be overloaded
        return None

    def generate_label(self, *argv):
        return None

    def generate_gt_bbx(self, data_dict):
        """
        The base postprocessor will generate 3d groundtruth bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []
        # used to avoid repetitive bounding box
        object_id_list = []
        unified_id_list = []  # 使用 ass_id 关联后的统一ID列表

        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space.
            # the transformation matrix for gt should always be based on
            # current timestamp (object transformation matrix is for
            # late fusion only since other fusion method already did
            #  the transformation in the preprocess)
            transformation_matrix = cav_content['transformation_matrix'] \
                if 'gt_transformation_matrix' not in cav_content \
                else cav_content['gt_transformation_matrix']

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            # 获取 vehicles 信息以访问 ass_id
            vehicles_info = cav_content.get('vehicles', {})
            
            # convert center to corner
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)

            # append the corresponding ids
            object_id_list += object_ids
            
            # 根据 ass_id 生成统一ID用于去重
            for obj_id in object_ids:
                # 检查该物体是否在 vehicles_info 中有 ass_id 信息
                if str(obj_id) in vehicles_info and 'ass_id' in vehicles_info[str(obj_id)]:
                    ass_id = vehicles_info[str(obj_id)]['ass_id']
                    if ass_id != -1:
                        # 使用 ass_id 作为统一ID（CP端关联到Ego端的ID）
                        unified_id_list.append(ass_id)
                    else:
                        # ass_id 为 -1，使用原始ID，但加上 cav_id 偏移避免不同CAV间的ID冲突
                        unified_id = obj_id + (1000 * hash(str(cav_id)) % 1000)
                        unified_id_list.append(unified_id)
                else:
                    # 没有 vehicles_info 或 ass_id 信息，使用原始ID
                    # 对于 ego CAV，通常 cav_id 为 'ego'，保持原始ID
                    if cav_id == 'ego':
                        unified_id_list.append(obj_id)
                    else:
                        # 对于其他 CAV，加上 cav_id 偏移
                        unified_id = obj_id + (1000 * hash(str(cav_id)) % 1000)
                        unified_id_list.append(unified_id)

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        # some of the bbx may be repetitive, use the unified id list to filter
        # TODO: gt 列表框改称 k-v 字典
        # gt_object_id_list = sorted(set(unified_id_list))
        # gt_box3d_selected_indices = \
        #     [unified_id_list.index(x) for x in gt_object_id_list]
        gt_box3d_selected_indices = \
            [unified_id_list.index(x) for x in set(unified_id_list)]
        # print(gt_box3d_selected_indices, gt_box3d_selected_indices1)
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

        # TODO: 加一个返回相应的 id list
        selected_object_ids = [object_id_list[i] for i in gt_box3d_selected_indices]
        filtered_object_ids = [selected_object_ids[i] for i in range(len(selected_object_ids)) if mask[i]]

        return gt_box3d_tensor, filtered_object_ids

    def generate_cp_gt_bbx(self, data_dict):
        """
        Generate CP ground truth bounding boxes using only CP's local coordinate system.
        Returns CP's local object IDs (the actual keys from vehicles dict).
        """
        cav_content = data_dict['1']
    
        # Get vehicles info to access local object IDs
        vehicles_info = cav_content.get('vehicles', {})
        if not vehicles_info:
            device = torch.device('cpu')
            if 'object_bbx_center' in cav_content:
                device = cav_content['object_bbx_center'].device
            return torch.empty(0, 8, 3, device=device), []
        
        # Get the raw data
        object_bbx_center = cav_content['object_bbx_center']  # (1, Max_N, 7)
        object_bbx_mask = cav_content['object_bbx_mask']     # (1, Max_N,)
        
        # Remove batch dimension
        if len(object_bbx_center.shape) == 3:
            object_bbx_center = object_bbx_center.squeeze(0)
        if len(object_bbx_mask.shape) == 2:
            object_bbx_mask = object_bbx_mask.squeeze(0)
        
        # Convert mask to numpy for boolean indexing
        if isinstance(object_bbx_mask, torch.Tensor):
            valid_mask = (object_bbx_mask == 1).cpu().numpy()
            object_bbx_center_valid = object_bbx_center[valid_mask]
        else:
            valid_mask = (object_bbx_mask == 1)
            object_bbx_center_valid = object_bbx_center[valid_mask]
        
        if object_bbx_center_valid.shape[0] == 0:
            device = object_bbx_center.device if hasattr(object_bbx_center, 'device') else torch.device('cpu')
            return torch.empty(0, 8, 3, device=device), []
        
        # The key insight: use vehicles_info keys as the local object IDs
        # The order should match the valid objects
        local_object_ids = list(vehicles_info.keys())
        
        # Filter to only include valid objects (same count as valid boxes)
        valid_local_object_ids = local_object_ids[:object_bbx_center_valid.shape[0]]
        
        # Convert to corner format
        if isinstance(object_bbx_center_valid, torch.Tensor):
            object_bbx_center_valid_np = object_bbx_center_valid.cpu().numpy()
        else:
            object_bbx_center_valid_np = object_bbx_center_valid
            
        gt_box3d_corner = box_utils.boxes_to_corners_3d(
            object_bbx_center_valid_np, 
            self.params['order']
        )
        
        gt_box3d_tensor = torch.from_numpy(gt_box3d_corner).to(object_bbx_center.device)
        
        # Range filtering
        mask_within_range = box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
        gt_box3d_tensor = gt_box3d_tensor[mask_within_range]
        filtered_object_ids = [int(valid_local_object_ids[i]) for i in range(len(valid_local_object_ids)) if mask_within_range[i]]
        
        return gt_box3d_tensor, filtered_object_ids

    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        TODO: 看看 car id 的问题

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.

        reference_lidar_pose : np.ndarray
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        from opencood.data_utils.datasets import GT_RANGE
        # GT_RANGE = [-100, -40, -5, 100, 40, 3]

        tmp_object_dict = {}
        for cav_content in cav_contents:
            tmp_object_dict.update(cav_content['params']['vehicles'])
            cav_id = cav_content['cav_id']

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else GT_RANGE

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'])

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_content) in enumerate(output_dict.items()):
            object_bbx = object_content['coord']
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            if object_content['ass_id'] != -1:
                object_ids.append(object_content['ass_id'])
            else:
                object_ids.append(object_id + 100 * cav_id)

        return object_np, mask, object_ids
