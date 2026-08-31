import torch
import torch_scatter
import torch.nn as nn
import numpy as np
from network.torchsparse_compat import sparse_tensor

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = torch.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return torch.stack((rho, phi, input_xyz[:, 2]), axis=1)


class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list):
        super(voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]

        if not hasattr(self, '_coors_min') or self._coors_min.device != pc.device:
            coors_min = [self.coors_range_xyz[0][0], self.coors_range_xyz[1][0], self.coors_range_xyz[2][0]]
            coors_max = [self.coors_range_xyz[0][1], self.coors_range_xyz[1][1], self.coors_range_xyz[2][1]]
            self._coors_min = torch.tensor(coors_min, dtype=pc.dtype, device=pc.device)
            self._coors_range = torch.tensor([coors_max[0] - coors_min[0], coors_max[1] - coors_min[1], coors_max[2] - coors_min[2]], dtype=pc.dtype, device=pc.device)

        for idx, scale in enumerate(self.scale_list):
            spatial_shape_scale = torch.tensor([
                np.ceil(self.spatial_shape[0] / scale),
                np.ceil(self.spatial_shape[1] / scale),
                np.ceil(self.spatial_shape[2] / scale)
            ], dtype=pc.dtype, device=pc.device)
            
            xyz_idx = (spatial_shape_scale * (pc - self._coors_min) / self._coors_range).long()
            bxyz_indx = torch.cat([data_dict['batch_idx'].unsqueeze(-1), xyz_idx], dim=-1)

            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32)
            }
        return data_dict


class voxelization_fixvs(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list, voxel_size):
        super(voxelization_fixvs, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz
        self.voxel_size = voxel_size
 
    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]

        for idx, scale in enumerate(self.scale_list):
            xyz_indx = torch.floor(pc / (self.voxel_size * scale))
            xyz_indx -= torch.min(xyz_indx, 0)[0]
            xyz_indx = xyz_indx.long()
            sparse_shape = torch.add(torch.max(xyz_indx, dim=0).values, 1).tolist()[::-1]
            bxyz_indx = torch.cat([data_dict['batch_idx'].unsqueeze(-1), xyz_indx], dim=-1)
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32),
                'spatial_shape': sparse_shape
            }
        return data_dict


class voxel_3d_generator(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape):
        super(voxel_3d_generator, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(in_channels),

            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Linear(out_channels, out_channels),
        )

    def prepare_input(self, point, grid_ind, inv_idx, normal=None):
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        if not hasattr(self, '_coors_range_xyz_tensor') or self._coors_range_xyz_tensor.device != point.device:
            self._coors_range_xyz_tensor = torch.tensor(self.coors_range_xyz, dtype=point.dtype, device=point.device)
            self._cur_grid_size_tensor = torch.tensor(self.spatial_shape, dtype=point.dtype, device=point.device)
            self._crop_range = self._coors_range_xyz_tensor[:, 1] - self._coors_range_xyz_tensor[:, 0]
            self._intervals = self._crop_range / self._cur_grid_size_tensor
            self._coors_min = self._coors_range_xyz_tensor[:, 0]

        voxel_centers = grid_ind * self._intervals + self._coors_min
        center_to_point = point[:, :3] - voxel_centers
        pc_feature = torch.cat((point, nor_pc, center_to_point, normal), dim=1)

        return pc_feature

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv'],
            data_dict['normal']
        )
        pt_fea = self.PPmodel(pt_fea)

        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)
        data_dict['sparse_tensor'] = sparse_tensor(
            features, data_dict['scale_1']['coors'],
            np.int32(self.spatial_shape)[::-1].tolist(), data_dict['batch_size'])


        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']

        return data_dict

class voxel_3d_generator_fixvs(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape, voxel_size):
        super(voxel_3d_generator_fixvs, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.voxel_size = voxel_size
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(in_channels),

            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Linear(out_channels, out_channels),
        )

    def prepare_input(self, point, grid_ind, inv_idx, normal=None):
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        min_volume_space = torch.floor(torch.min(point[:, :3], 0)[0])
        voxel_centers = grid_ind * self.voxel_size + min_volume_space
        center_to_point = point[:, :3] - voxel_centers

        pc_feature = torch.cat((point, nor_pc, center_to_point, normal), dim=1)

        return pc_feature

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv'],
            data_dict['normal']
        )
        pt_fea = self.PPmodel(pt_fea)

        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)
        data_dict['sparse_tensor'] = sparse_tensor(
            features, data_dict['scale_1']['coors'],
            data_dict['scale_1']['spatial_shape'], data_dict['batch_size'])

        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']
        data_dict['spatial_shape'] = data_dict['scale_1']['spatial_shape']

        return data_dict
