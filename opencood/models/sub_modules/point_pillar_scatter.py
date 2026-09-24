import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']
        assert self.nz == 1

    def forward(self, batch_dict):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict[
            'voxel_coords']
        
        # 处理空点云的情况（某些 CAV 可能没有有效数据）
        if coords.numel() == 0 or pillar_features.numel() == 0:
            # 返回空的 spatial features，避免后续计算出错
            batch_dict['spatial_features'] = torch.zeros(
                1,  # batch_size=1 for nofusion
                self.num_bev_features * self.nz,
                self.ny,
                self.nx,
                dtype=pillar_features.dtype if pillar_features.numel() > 0 else torch.float32,
                device=pillar_features.device if pillar_features.numel() > 0 else 'cpu'
            )
            return batch_dict
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]

            indices = this_coords[:, 1] + \
                      this_coords[:, 2] * self.nx + \
                      this_coords[:, 3]
            indices = indices.type(torch.long)

            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features

        return batch_dict

