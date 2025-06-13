import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.layers import to_2tuple, trunc_normal_
import matplotlib.pyplot as plt
from PIL import Image


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size=[9, 9], n_heads=6, n_head_channels=16, n_groups=2,
            stride=2, use_pe=False, dwc_pe=False,
            no_off=False, fixed_pe = False, ksize=3, log_cpb=False
    ):

        super().__init__()
        self.dwc_pe = dwc_pe # 是否使用深度可分离卷积的位置编码
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5 # 缩放因子，用于缩放查询向量
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size # 查询向量大小
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads # 总通道数
        self.n_groups = n_groups # 组的数量，用于分组卷积
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe # 是否使用位置编码
        self.fixed_pe = fixed_pe # 是否使用固定位置编码
        self.no_off = no_off # 是否禁用偏移
        self.offset_range_factor = 1 # 偏移范围因子
        self.ksize = ksize
        self.log_cpb = log_cpb # 是否记录位置编码
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0 # 填充大小，确保输出大小一致

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # 线性层
        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(0.1, inplace=False)
        self.attn_drop = nn.Dropout(0.1, inplace=False)

        # 相对位置编码表，用于增强模型性能和表示能力
        if self.use_pe and not self.no_off:  # 使用位置编码，不禁用偏移
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w))
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=False),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1))
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device): # 生成参考点坐标

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # (B * g, H_key, W_key, 2)

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device): # 生成查询点的网格坐标

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off: # 禁用偏置
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        # if self.use_pe and (not self.no_off):
        #
        #     if self.dwc_pe:
        #         residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
        #                                                                       H * W)
        #     elif self.fixed_pe:
        #         rpe_table = self.rpe_table
        #         attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        #         attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
        #     elif self.log_cpb:
        #         q_grid = self._get_q_grid(H, W, B, dtype, device)
        #         displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
        #                                                                                            n_sample,
        #                                                                                            2).unsqueeze(1)).mul(4.0)  # d_y, d_x [-8, +8]
        #         displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
        #         attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
        #         attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
        #     else:
        #         rpe_table = self.rpe_table # (n_head, q_h*2-1, q_w*2-1)
        #         rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1) # (B, n_head, q_h, q_w)
        #         q_grid = self._get_q_grid(H, W, B, dtype, device) # (B * g, H, W, 2)
        #         displacement = (
        #                     q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
        #         attn_bias = F.grid_sample(
        #             input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
        #                                    g=self.n_groups),
        #             grid=displacement[..., (1, 0)],
        #             mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns
        #
        #         attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
        #         attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        # if self.use_pe and self.dwc_pe:
        #     out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        # return y
        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class LayerNormProxy(nn.Module): # 层归一化

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # 定义Sobel核在x方向上
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        # 定义Sobel核在y方向上
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.sobel_x = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y, requires_grad=False)

    def forward(self, x):
        # 确保输入数据是4维张量 (batch_size, channels, height, width)
        if x.dim()!= 4:
            raise ValueError('Input tensor must be 4 - dimensional')
        # 在x方向上进行卷积
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        # 在y方向上进行卷积
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        # 计算梯度幅值
        grad_magnitude = torch.sqrt(grad_x ** 2+grad_y ** 2)
        # 计算梯度方向
        grad_direction = torch.atan2(grad_y, grad_x)
        return grad_magnitude, grad_direction

def min_max_normalization(magnitude):
    min_val = torch.min(magnitude)
    max_val = torch.max(magnitude)
    normalized_magnitude = (magnitude - min_val) / (max_val - min_val)
    return normalized_magnitude

def binarize_grad_magnitude(grad_magnitude, threshold=0.5):
    # 创建一个布尔型张量，梯度幅值大于阈值的位置为 True，其余为 False
    binary_mask = grad_magnitude > threshold
    # 将布尔型张量转换为 0 和 1 的张量
    binary_tensor = binary_mask.float()
    return binary_tensor


class OAMoudle(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OAMoudle, self).__init__()

        self.inc = in_channels # 为4的倍数
        self.ouc = out_channels
        self.n_groups = 2 # 组的数量用于分组卷积
        self.n_group_channels = self.inc // self.n_groups
        self.n_heads = 4
        self.n_group_heads = self.n_heads // self.n_groups
        self.kernei_size = 3
        self.n_head_channels = in_channels//self.n_heads
        self.scale = self.n_head_channels ** -0.5  # 缩放因子，用于缩放查询向量
        self.offset_range_factor = 1  # 偏移范围因子

        self.sobel_filter = SobelFilter()
        self.sobel_conv = nn.Conv2d(self.n_group_channels, 1, kernel_size=1, stride=1, padding=0)

        # 线性层
        self.proj_q = nn.Conv2d(self.inc, self.inc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.inc, self.inc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.inc, self.inc, kernel_size=1, stride=1, padding=0)

        self.proj_out = nn.Conv2d(self.inc, self.inc, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(0.1, inplace=False)
        self.attn_drop = nn.Dropout(0.1, inplace=False)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kernei_size, 1, 1, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.rpe_table = nn.Conv2d(self.inc, self.inc, kernel_size=3, stride=1, padding=1, groups=self.inc)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device): # 生成参考点坐标

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # (B * g, H_key, W_key, 2)

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device): # 生成查询点的网格坐标

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device) # (B * g, H_key, W_key, 2)

        sobel_x, _ = self.sobel_filter(self.sobel_conv(q_off))
        # normalized_magnitude = min_max_normalization(sobel_x)
        # print('归一化后的梯度幅值 (最小-最大归一化):', normalized_magnitude)

        binary_tensor = binarize_grad_magnitude(sobel_x, threshold=0.5)
        binary_tensor = einops.rearrange(binary_tensor, 'b p h w -> b h w p')
        reference_sobel = reference * binary_tensor

        # 位置编码
        pos = offset + reference_sobel

        # 加入位置编码修改输入特征，生成新key和value
        x_sampled = F.grid_sample(input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid= pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        # 位置编码dwc
        residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos, reference

if __name__ == '__main__':

    x = torch.load('../patch_images/9x9tensor.pt').cpu() # （512， 64， 9， 9）
    input = x.cuda()
    # B,C,H,W = 8, 96, 9, 9
    # input = torch.randn(B,C,H,W)

    dat_net = OAMoudle(64, 64).cuda()
    y,pos,reference = dat_net(input)

    # 将张量转换为numpy数组
    y_np = y.detach().cpu().numpy()
    pos_np = pos.detach().cpu().numpy()
    reference_np = reference.detach().cpu().numpy()

    # 创建子图
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    # 显示张量 input
    axs[0].imshow(x[0,0], cmap='viridis')
    axs[0].set_title('Tensor input')

    # 显示张量 y
    yyy = y_np[0,0]
    axs[1].imshow(yyy, cmap='viridis')
    axs[1].set_title('Tensor y')

    # 显示张量 pos
    axs[2].imshow(pos_np[0,:,:,1], cmap='viridis')
    axs[2].set_title('Tensor pos')

    # 显示张量 reference
    axs[3].imshow(reference_np[0,:,:,1], cmap='viridis')
    axs[3].set_title('Tensor reference')

    # 调整布局
    plt.tight_layout()

    # 显示图像
    plt.show()


    print(y.shape)
    # print(pos, reference)



