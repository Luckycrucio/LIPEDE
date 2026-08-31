import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsparse import nn as spnn
from network.torchsparse_compat import SparseConv3d, sparse_tensor



# ============================================================
# PTv3-style spatial serialization utilities
# ============================================================

def morton_encode(coords, depth):
    """
    3D Morton / Z-order encoding.

    coords:
        [N, 3] integer coordinates

    Returns:
        [N] int64 serialization code
    """

    x = coords[:, 0].long()
    y = coords[:, 1].long()
    z = coords[:, 2].long()

    code = torch.zeros_like(x)

    for bit in range(depth):
        code |= ((x >> bit) & 1) << (3 * bit)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit + 2)

    return code


def serialize_z(coords, batch_idx, transformed=False):
    """
    PTv3-style Z-order serialization.

    Coordinates are normalized independently per batch.
    """

    coords = coords.long()
    batch_idx = batch_idx.long()

    n = coords.shape[0]

    if n == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=coords.device,
        )

    normalized = torch.empty_like(coords)

    num_batches = int(batch_idx.max()) + 1

    max_coord = 0

    # --------------------------------------------------------
    # Normalize each batch independently.
    # --------------------------------------------------------

    for b in range(num_batches):

        mask = batch_idx == b

        if not mask.any():
            continue

        c = coords[mask]

        c_min = c.min(dim=0).values
        c = c - c_min

        normalized[mask] = c

        max_coord = max(
            max_coord,
            int(c.max().item()),
        )

    # Number of bits required.
    depth = max(
        1,
        int(max_coord).bit_length(),
    )

    if depth > 20:
        raise RuntimeError(
            f"Coordinate depth={depth} is too large for "
            "64-bit Morton serialization."
        )

    if transformed:

        # PTv3-style transformed traversal.
        #
        # Reversing one axis gives another traversal with
        # similar locality but different token adjacency.

        normalized = normalized.clone()

        max_val = (1 << depth) - 1

        normalized[:, 0] = (
            max_val - normalized[:, 0]
        )

    code = morton_encode(
        normalized,
        depth,
    )

    return code


@torch.no_grad()
def ptv3_serialize(
    coords,
    batch_idx,
    mode="z",
):
    """
    Serialize points in a PTv3-like fashion.

    Important:
        batch_idx is part of the ordering, so points from
        different samples can never enter the same RWKV window.

    Returns:
        order
        inverse
    """

    if mode == "z":

        code = serialize_z(
            coords,
            batch_idx,
            transformed=False,
        )

    elif mode == "z-trans":

        code = serialize_z(
            coords,
            batch_idx,
            transformed=True,
        )

    else:
        raise NotImplementedError(
            f"Serialization mode '{mode}' is not implemented "
            "in this self-contained version. "
            "Use 'z' or 'z-trans'."
        )

    # --------------------------------------------------------
    # Make serialization batch-aware.
    # --------------------------------------------------------

    code_max = code.max()

    global_code = (
        batch_idx.long() * (code_max + 1)
        + code
    )

    order = torch.argsort(
        global_code,
        stable=True,
    )

    # inverse[original_index] = serialized_index
    inverse = torch.empty_like(order)

    inverse[order] = torch.arange(
        order.numel(),
        device=order.device,
    )

    return order, inverse


# ============================================================
# Local-window RWKV
# ============================================================

class RWKVBlock(nn.Module):
    """
    PTv3-style local-window RWKV.

    Pipeline:

        input points
              |
              v
        spatial serialization
              |
              v
        local windows
              |
              v
        token shift
              |
              v
        local linear RWKV
              |
              v
        restore original point order

    Unlike the original implementation, the KV state is NOT
    computed over the entire batch.

    Each spatial window has its own KV state.

    Parameters
    ----------
    dim:
        Feature dimension.

    window_size:
        Number of serialized points per local RWKV window.

    serialization:
        "z" or "z-trans".

    Note
    ----
    This is deliberately padding-free. The final window of a
    batch can contain fewer than window_size points.
    """

    def __init__(
        self,
        dim,
        window_size=1024,
        serialization="z",
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.serialization = serialization

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        self.ln = nn.LayerNorm(dim)

        # ----------------------------------------------------
        # RWKV time mixing
        # ----------------------------------------------------

        self.time_mix_k = nn.Parameter(
            torch.rand(1, dim)
        )

        self.time_mix_v = nn.Parameter(
            torch.rand(1, dim)
        )

        self.time_mix_r = nn.Parameter(
            torch.rand(1, dim)
        )

        # ----------------------------------------------------
        # Projections
        # ----------------------------------------------------

        self.key = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.value = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.receptance = nn.Linear(
            dim,
            dim,
            bias=True,
        )

        self.output = nn.Linear(
            dim,
            dim,
            bias=True,
        )

    # ========================================================
    # Token shift
    # ========================================================

    @staticmethod
    def token_shift(
        x,
        window_id,
    ):
        """
        Shift only inside the same local window.

        The first token of every window receives a zero shift.

        This is important.

        A naive x[:-1] shift would allow:

            window 0 last token
                       |
                       v
            window 1 first token

        to communicate.

        We explicitly prevent that.
        """

        x_shift = torch.zeros_like(x)

        if x.shape[0] <= 1:
            return x_shift

        same_window = (
            window_id[1:]
            == window_id[:-1]
        )

        x_shift[1:] = torch.where(
            same_window[:, None],
            x[:-1],
            torch.zeros_like(x[:-1]),
        )

        return x_shift

    # ========================================================
    # Construct local windows
    # ========================================================

    @staticmethod
    def make_windows(
        batch_idx,
        window_size,
    ):
        """
        Construct a unique window ID for every serialized point.

        Points are assumed to already be sorted as:

            batch 0:
                window 0
                window 1
                ...

            batch 1:
                window 0
                window 1
                ...

        Returns
        -------
        window_id:
            [N]

        local_position:
            position inside the window
        """

        n = batch_idx.numel()

        if n == 0:
            empty = torch.empty(
                0,
                dtype=torch.long,
                device=batch_idx.device,
            )

            return empty, empty

        # ----------------------------------------------------
        # Position within each batch.
        # ----------------------------------------------------

        # Since serialization guarantees batch-contiguous
        # ordering, we can calculate local positions using
        # cumulative counts.

        batch_change = torch.ones(
            n,
            dtype=torch.bool,
            device=batch_idx.device,
        )

        if n > 1:
            batch_change[1:] = (
                batch_idx[1:]
                != batch_idx[:-1]
            )

        batch_starts = torch.nonzero(
            batch_change,
            as_tuple=False,
        ).flatten()

        # Number of points before current batch.
        batch_number = torch.cumsum(
            batch_change.long(),
            dim=0,
        ) - 1

        start_for_point = batch_starts[
            batch_number
        ]

        position = (
            torch.arange(
                n,
                device=batch_idx.device,
            )
            - start_for_point
        )

        # ----------------------------------------------------
        # Window inside each batch.
        # ----------------------------------------------------

        window_number = (
            position // window_size
        )

        # Need globally unique window IDs.
        #
        # Maximum possible number of windows per batch is:
        #
        # ceil(N / window_size)
        #

        max_windows = (
            (n + window_size - 1)
            // window_size
        )

        window_id = (
            batch_number * max_windows
            + window_number
        )

        return window_id, position

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, input_):

        if len(input_) == 4:

            x, batch_idx, coords, batch_size = input_

        else:

            x, batch_idx, coords = input_

            batch_size = (
                int(batch_idx.max().item()) + 1
                if batch_idx.numel() > 0
                else 0
            )

        # ----------------------------------------------------
        # Residual
        # ----------------------------------------------------

        residual = x

        # ----------------------------------------------------
        # LayerNorm
        # ----------------------------------------------------

        x = self.ln(x)

        if x.shape[0] == 0:
            return (
                residual,
                batch_idx,
                coords,
                batch_size,
            )

        # ====================================================
        # 1. Spatial serialization
        # ====================================================

        order, inverse = ptv3_serialize(
            coords=coords,
            batch_idx=batch_idx,
            mode=self.serialization,
        )

        x = x[order]

        sorted_batch_idx = (
            batch_idx[order]
        )

        # ====================================================
        # 2. Construct local windows
        # ====================================================

        window_id, local_position = (
            self.make_windows(
                sorted_batch_idx,
                self.window_size,
            )
        )

        # ====================================================
        # 3. Token shift
        # ====================================================

        x_shift = self.token_shift(
            x,
            window_id,
        )

        # ====================================================
        # 4. Time mixing
        # ====================================================

        xk = (
            x * self.time_mix_k
            +
            x_shift * (
                1.0 - self.time_mix_k
            )
        )

        xv = (
            x * self.time_mix_v
            +
            x_shift * (
                1.0 - self.time_mix_v
            )
        )

        xr = (
            x * self.time_mix_r
            +
            x_shift * (
                1.0 - self.time_mix_r
            )
        )

        # ====================================================
        # 5. RWKV projections
        # ====================================================

        k = F.softplus(
            self.key(xk)
        )

        v = self.value(xv)

        r = torch.sigmoid(
            self.receptance(xr)
        )

        # ====================================================
        # 6. LOCAL linear attention
        # ====================================================

        # Original implementation:
        #
        #     KV = sum(k * v, batch)
        #     K  = sum(k, batch)
        #
        # We replace "batch" with "window".
        #
        # Therefore:
        #
        #     KV_w = sum_{i in window w} k_i v_i
        #
        #     K_w  = sum_{i in window w} k_i
        #
        # and every point only sees its local window.
        # ====================================================

        kv = torch_scatter.scatter_sum(
            k * v,
            window_id,
            dim=0,
        )

        k_sum = torch_scatter.scatter_sum(
            k,
            window_id,
            dim=0,
        )

        attn = (
            kv[window_id]
            /
            k_sum[window_id].clamp(
                min=1e-6
            )
        )

        # ====================================================
        # 7. Receptance
        # ====================================================

        out = r * attn

        out = self.output(out)

        # ====================================================
        # 8. Restore original ordering
        # ====================================================

        out = out[inverse]

        # ====================================================
        # 9. Residual
        # ====================================================

        out = residual + out

        return (
            out,
            batch_idx,
            coords,
            batch_size,
        )

class SparseBasicBlock(nn.Module):
    def __init__(self, large_kernel, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = nn.Sequential(
            SparseConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            spnn.BatchNorm(out_channels),
        )
        self.layers = nn.Sequential(
            SparseConv3d(in_channels, out_channels, kernel_size=large_kernel, indice_key=indice_key, bias=False),
            spnn.BatchNorm(out_channels),
            spnn.LeakyReLU(0.1),
            SparseConv3d(out_channels, out_channels, kernel_size=large_kernel, indice_key=indice_key, bias=False),
            spnn.BatchNorm(out_channels),
        )

        self.weight_initialization(self.layers) # Initalize weights

    def weight_initialization(self, layers):
        for m in layers:
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.layers_in(x)
        output = self.layers(x)
        output.F = F.leaky_relu(output.F + identity.F, 0.1)
        return output

# -----------------------------
# RWKV Point Encoder
# -----------------------------
class RWKVPointEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale, rwkv_layers):
        super().__init__()
        self.scale = scale
        self.rwkv_layers = rwkv_layers
        print("creating RWKVPointEncoder with", self.rwkv_layers, "rwkv layers")

        # Identity branch (full resolution)
        self.identity_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )

        # Downsample branch before RWKV
        self.pre_proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )

        # RWKV attention on downsampled tokens
        rwkv_blocks = [RWKVBlock(out_channels) for _ in range(self.rwkv_layers)]
        self.rwkv = nn.Sequential(*rwkv_blocks)
        #try:
        #    self.rwkv = torch.compile(self.rwkv, dynamic=True)
        #except Exception as e:
        #    print(f"torch.compile of rwkv blocks failed: {e}")

        # Final fusion
        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels),
        )

    @staticmethod
    def downsample(coors, p_fea, scale=2):
        batch = coors[:, 0:1]
        coors = coors[:, 1:] // scale
        merged = torch.cat([batch, coors], 1)

        unique, inv = torch.unique(merged, return_inverse=True, dim=0)
        down = torch_scatter.scatter_mean(p_fea, inv, dim=0)

        return down, inv, unique

    def forward(self, features, data_dict):
        """
        features: [N_points, C]
        """

        # -------------------------
        # Identity branch (no downsample)
        # -------------------------
        identity = self.identity_proj(features)

        # -------------------------
        # Downsample points
        # -------------------------
        down_feat, inv, new_coors = self.downsample(
            data_dict['coors'], features, self.scale
        )

        # Project before RWKV
        down_feat = self.pre_proj(down_feat)

        # -------------------------
        # RWKV attention on voxel tokens
        # -------------------------
        down_batch_idx = new_coors[:, 0]  # [N_down]
        coords_no_batch = new_coors[:, 1:]  # remove batch column
        
        batch_size = data_dict['batch_size']
        input_ = (down_feat, down_batch_idx.long(), coords_no_batch, batch_size)
        (down_feat, _, _, _) = self.rwkv(input_)
        
        down_feat_full = down_feat[inv]
        # -------------------------
        # Fusion
        # -------------------------
        fused = torch.cat([identity, down_feat_full], dim=1)
        fused = self.layer_out(fused)

        # -------------------------
        # Aggregate to voxel level
        # -------------------------
        v_feat = torch_scatter.scatter_mean(
            fused[data_dict['coors_inv']],
            data_dict[f'scale_{self.scale}']['coors_inv'],
            dim=0
        )

        # -------------------------
        # Update data_dict for next scale
        # -------------------------
        data_dict['coors'] = data_dict[f'scale_{self.scale}']['coors']
        data_dict['coors_inv'] = data_dict[f'scale_{self.scale}']['coors_inv']
        data_dict['full_coors'] = data_dict[f'scale_{self.scale}']['full_coors']

        return v_feat

class SPVBlock(nn.Module):
    def __init__(self, large_kernel, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape, rwkv_layers):
        super(SPVBlock, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.v_enc = nn.Sequential(
            SparseBasicBlock(large_kernel, in_channels, out_channels, self.indice_key),
            SparseBasicBlock(large_kernel, out_channels, out_channels, self.indice_key),
        )
        #self.p_enc = point_encoder(in_channels, out_channels, scale)
        self.p_enc = RWKVPointEncoder(in_channels, out_channels, scale, rwkv_layers)
        print(f"created RWKVPointEncoder with in channels {in_channels} out channels {out_channels} and scale {scale} ")

    def forward(self, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv']

        # voxel encoder
        v_fea = self.v_enc(data_dict['sparse_tensor'])
        data_dict['layer_{}'.format(self.layer_id)] = {}
        data_dict['layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.F
        data_dict['layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['full_coors']
        v_fea_inv = torch_scatter.scatter_mean(v_fea.F[coors_inv_last], coors_inv, dim=0)

        # point encoder
        p_fea = self.p_enc(
            features=data_dict['sparse_tensor'].F + v_fea.F,
            data_dict=data_dict
        )

        # fusion and pooling
        data_dict['sparse_tensor'] = sparse_tensor(
            p_fea + v_fea_inv, data_dict['coors'], self.spatial_shape,
            data_dict['batch_size'])

        

        return p_fea[coors_inv]



