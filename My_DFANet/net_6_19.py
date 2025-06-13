
import torch.nn.functional as F
from My_DFANet.module_files.Attention import *
from My_DFANet.module_files.offset_module import DAttentionBaseline, OAMoudle
from torch.nn.parameter import Parameter
from skimage.metrics import structural_similarity as ssim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.gn = nn.GroupNorm(num_groups=4, num_channels=out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.ELU = nn.ELU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.gn(x)
        # x = self.relu(x)
        x = self.leaky_relu(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.conv1x1 = nn.Conv2d(in_features, 1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.leaky_relu(x)
        return x

# 深度可分离卷积
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class new_feature_align(nn.Module):
    def __init__(self, in_channel, out_channel, lidar_channel):
        super(new_feature_align, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.middle_channel = out_channel//2 # middle channel
        self.lidar_channel = lidar_channel

        self.conv1x1_h = nn.Conv2d(in_channel, in_channel, 1)
        self.conv1x1_l = nn.Conv2d(lidar_channel, lidar_channel, 1)
        self.deconv_h = DepthwiseSeparableConv2d(in_channel, in_channel, 1)
        self.deconv_l = DepthwiseSeparableConv2d(lidar_channel, 1, 1)
        self.deconv_h_channel = DepthwiseSeparableConv2d(in_channel, 1, 1)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_pool =nn.Conv2d(in_channel, 2, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.mlp1 = Mlp(4*9*9, 4*9*9, 1*9*9) # weight(b,4,h,w)
        self.mlp2 = Mlp(4*5*5, 4*5*5, 1*5*5)  # weight(b,4,h,w)

        # self.mlp1 = Mlp(4*25*25, 4*25*25, 1*25*25) # weight(b,4,h,w)
        # self.mlp2 = Mlp(4*13*13, 4*13*13, 1*13*13)  # weight(b,4,h,w)

        self.bconv2d = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1)
        self.channel_att = ChannelAttention(in_channel)
        self.spectral_att = SpatialAttention()

    def forward(self, x, y, index=None):

        b,c,h,w = x.size()
        x1 = self.deconv_h(self.conv1x1_h(x))
        y1 = self.deconv_l(self.conv1x1_l(y))

        x1_channel = self.deconv_h_channel(x1)
        cos_ = F.cosine_similarity(x1, y1, dim=1)
        cos_ = torch.unsqueeze(cos_, dim=1)

        # 特征补充:cos为-1表示完全不相似，为1表示完全相似
        negative_cos_ = torch.where(cos_ < 0, cos_, torch.tensor(0.0))
        positive_cos_ = torch.where(cos_ > 0, cos_, torch.tensor(0.0))
        # x2 = x1 + (-negative_cos_) * y1
        # x1 = x1 + (-negative_cos_) * y1

        x2 = x1 + 0.5*(1 - cos_) * y1
        xap = torch.mean(x2, dim=1, keepdim=True)
        xmp = torch.max(x2, dim=1, keepdim=True)[0]

        # xap = self.conv1x1_pool(self.avgpool(x2))
        # xmp = self.conv1x1_pool(self.maxpool(x2))
        xc = torch.cat((xap, self.conv1x1_pool(x2)),dim=1)
        xc = torch.cat((xc, xmp), dim=1)#(b,3,h,w)
        xc = xc.reshape(b,-1)
        if index == None:
            xw = self.mlp1(xc)
        else:
            xw = self.mlp2(xc)
        xw = xw.reshape(b,1,h,w)
        x3 = xw * x2

        # xout = x3
        xout = x3 + x1
        xout = self.bconv2d(xout)

        return xout

class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=1, device='cpu'):
        super(SpatialSoftmax, self).__init__()

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature).to(device)
        else:
            self.temperature = 1.

    def forward(self, feature):
        feature = feature.view(feature.shape[0], -1, feature.shape[1] * feature.shape[2])
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)

        return softmax_attention

class SpectralSoftmax(nn.Module):
    def __init__(self, temperature=1, device='cpu' ):
        super(SpectralSoftmax, self).__init__()
        self.dim = 1  # softmax 应用的维度, 默认为光谱维度


        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature).to(device)
        else:
            self.temperature = 1.

    def forward(self, feature):
        # feature 的形状假设为 [B, C, H, W]，其中 C 是光谱维度
        softmax_attention = F.softmax(feature / self.temperature, dim=self.dim)
        return softmax_attention


def at_gen(x1, x2):
    """
    x1 - previous encoder step feature map
    x2 - current encoder step feature map
    """
    at_gen_upsample = nn.Upsample(size=9, mode='bilinear', align_corners=False)
    at_gen_l2_loss = nn.MSELoss(reduction='mean')
    # G^2_sum
    sps = SpatialSoftmax(device=x1.device)

    if x1.size() != x2.size():
        x1 = x1.pow(2).sum(dim=1)
        x1 = sps(x1)
        x2 = x2.pow(2).sum(dim=1, keepdim=True)
        x2 = torch.squeeze(at_gen_upsample(x2), dim=1)
        x2 = sps(x2)
    else:
        x1 = x1.pow(2).sum(dim=1)
        x1 = sps(x1)
        x2 = x2.pow(2).sum(dim=1)
        x2 = sps(x2)

    loss = at_gen_l2_loss(x1, x2)
    return loss

def conv_gen(x1, x2):
    """
    x1 - previous encoder step feature map
    x2 - current encoder step feature map
    """
    at_gen_upsample = nn.Upsample(size=9, mode='bilinear', align_corners=False)
    at_gen_l2_loss = nn.MSELoss(reduction='mean')
    # G^2_sum
    sps = SpectralSoftmax(device=x1.device)

    if x1.size() != x2.size():
        x1 = x1.pow(2).sum(dim=1)
        x1 = sps(x1)
        x2 = x2.pow(2).sum(dim=1, keepdim=True)
        x2 = torch.squeeze(at_gen_upsample(x2), dim=1)
        x2 = sps(x2)
    else:
        x1 = x1.pow(2).sum(dim=1)
        x1 = sps(x1)
        x2 = x2.pow(2).sum(dim=1)
        x2 = sps(x2)

    loss = at_gen_l2_loss(x1, x2)
    return loss

class FAFMoudle(nn.Module):
    def __init__(self, in_channel, out_channel, lidar_channel):
        super(FAFMoudle, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.middle_channel = out_channel // 2  # middle channel
        self.lidar_channel = lidar_channel

        self.conv1x1_h = nn.Conv2d(in_channel, in_channel, 1)
        self.conv1x1_l = nn.Conv2d(lidar_channel, lidar_channel, 1)
        self.deconv_h = DepthwiseSeparableConv2d(in_channel, in_channel, 1)
        self.deconv_l = DepthwiseSeparableConv2d(lidar_channel, 1, 1)
        self.deconv_h_channel = DepthwiseSeparableConv2d(in_channel, 1, 1)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_pool = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.mlp1 = Mlp(4 * 9 * 9, 4 * 9 * 9, 1 * 9 * 9)  # weight(b,4,h,w)
        self.mlp2 = Mlp(4 * 5 * 5, 4 * 5 * 5, 1 * 5 * 5)  # weight(b,4,h,w)

        self.bconv2d = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1)

        self.conv1_1 = nn.Conv2d(in_channel, in_channel, 1)
        self.conv1_2 = nn.Conv2d(2, 2, 1)

        self.convh_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.convh_2 = nn.Conv2d(in_channel, in_channel*2//3, kernel_size=1, stride=1, padding=0)
        self.convh_3 = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0)
        self.convl_1 = nn.Conv2d(lidar_channel, in_channel*1//3, kernel_size=1, stride=1, padding=0)
        self.convl_2 = nn.Conv2d(lidar_channel, 1, kernel_size=1, stride=1, padding=0)
        self.convl_3 = nn.Conv2d(lidar_channel, 2, kernel_size=1, stride=1, padding=0)

        self.convh2_1 = nn.Conv2d(in_channel, in_channel*2//3, kernel_size=1, stride=1, padding=0)
        self.convl2_1 = nn.Conv2d(lidar_channel, in_channel*1//3, kernel_size=1, stride=1, padding=0)

        self.change_channel1 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)
        self.change_channel2 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)

        self.middle_change_channel = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0)

    def ssim_fun(self, image1, image2, index=None):
        image1 = image1.detach().cpu().numpy()
        image2 = image2.detach().cpu().numpy()
        if index == None:
            return ssim(image1, image2, data_range=1.0, win_size=7, channel_axis=1, full=True)[1]
        else:
            return ssim(image1, image2, data_range=1.0, win_size=5, channel_axis=1, full=True)[1]

    def forward(self, x, y, index=None):

        b, c, h, w = x.size()
        fuse_1 = self.convh_1(x)
        fuse_2 = self.conv1_1(torch.cat((self.convh_2(x),self.convl_1(y)), dim=1))
        fuse_3 = self.conv1_2(torch.cat((self.convh_3(x),self.convl_2(y)), dim=1))
        fuse_4 = self.convl_3(y)

        cor_1 = F.cosine_similarity(fuse_1, fuse_2, dim=1)
        cor_1 = torch.unsqueeze(cor_1, dim=1)
        ssim_value = self.ssim_fun(fuse_3, fuse_4, index)
        ssim_value = torch.tensor(ssim_value).to(device)

        fuse2_1 = fuse_1 + 0.5 * (1 - cor_1) * fuse_2
        fuse2_2 = fuse_4 + ssim_value * fuse_3

        fuse2_2 =self.change_channel1(fuse2_2)
        cor2_1 = F.cosine_similarity(fuse2_1, fuse2_2, dim=1)
        cor2_1 = torch.unsqueeze(cor2_1, dim=1)

        fuse3_1 = fuse2_1 + 0.5 * (1 - cor2_1 + self.change_channel1(ssim_value)) * fuse2_2

        xap = torch.mean(fuse2_2, dim=1, keepdim=True)
        xmp = torch.max(fuse2_2, dim=1, keepdim=True)[0]
        xc = torch.cat((xap, self.conv1x1_pool(fuse2_2)), dim=1)
        xc = torch.cat((xc, xmp), dim=1)  # (b,3,h,w)
        xc = xc.reshape(b, -1)
        if index == None:
            xweight = self.mlp1(xc)
        else:
            xweight = self.mlp2(xc)
        xweight = xweight.reshape(b, 1, h, w)
        fuse4_1 = xweight * fuse3_1

        xout = fuse4_1 + fuse3_1
        xout = self.bconv2d(xout)

        return xout


class DFANet(nn.Module):
    def __init__(self, config):
        super(DFANet, self).__init__()
        band = config['in_channels']
        classes = config['num_classes']
        dim = config['middle_dim']
        lidar_dim = config['in_channels_lidar']

        self.conv1x1 = nn.Conv2d(band, dim, 1)
        self.feature_align1 = new_feature_align(dim, dim, lidar_dim)

        self.channel_att = ChannelAttention(dim)
        self.spectral_att = SpatialAttention()

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.feature_align2 = new_feature_align(dim, dim, lidar_dim)
        self.basic_conv_l = BasicConv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1)
        self.basic_conv_r = BasicConv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1)

        self.flattend_conv1 = BasicConv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=1)
        self.flattend_conv2 = BasicConv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.cls1 = nn.Linear(dim, classes)
        self.cls2 = nn.Linear(dim, classes)

        # 通道分离
        self.channel_att_2 = ChannelAttention(dim//2)
        self.dat_net = DAttentionBaseline()

        self.offset_module1 = OAMoudle(dim, dim)
        self.offset_module2 = OAMoudle(dim, dim)

        self.correlation_m1 = FAFMoudle(dim, dim, lidar_dim)
        self.correlation_m2 = FAFMoudle(dim, dim, lidar_dim)

    def forward(self, x, y, pseudo_x=None):

        x = self.conv1x1(x)
        feat_x = self.relu(self.correlation_m1(x, y))
        # feat_x = self.relu(x)

        # 左分支
        # 偏离注意力
        feat_xl = feat_x
        off_feat = self.leaky_relu(self.offset_module1(feat_xl)[0])
        # off_feat_x2 = self.avgpool(off_feat)
        off_feat_x2 = self.maxpool(off_feat)
        off_feat_x2 = self.offset_module2(off_feat_x2)[0]
        # 反池化
        unpool_off = F.interpolate(off_feat_x2, size=9, mode='bilinear')
        # 跳连输出
        logit_l = unpool_off

        # 右分支
        # 通道不分离
        chan_featx = self.channel_att(feat_x) * feat_x
        chan_featx = self.leaky_relu(chan_featx)
        spec_featx = self.spectral_att(feat_x) * feat_x
        spec_featx = self.leaky_relu(spec_featx)
        feat_x2 = chan_featx + spec_featx

        # feat_x2_dowm = self.avgpool(feat_x2)
        feat_x2_dowm = self.maxpool(feat_x2)
        # 池化
        # y_pool = self.avgpool(y)
        y_pool = self.maxpool(y)

        feat_x3 = self.correlation_m2(feat_x2_dowm, y_pool, 1)
        # 反池化
        unpoolx = F.interpolate(feat_x3, size=9, mode='bilinear')
        # 跳连输出
        logit_r = unpoolx
        # logit_r = feat_x2

        # logit = dat_out
        logit_l = self.basic_conv_l(logit_l)
        logit_r = self.basic_conv_r(logit_r)

        xout_l = self.avg(logit_l).squeeze(-1).squeeze(-1)
        xout_l = self.cls1(xout_l)

        xout_r = self.avg(logit_r).squeeze(-1).squeeze(-1)
        xout_r = self.cls2(xout_r)

        logit = logit_r + logit_l
        xout = xout_r + xout_l
        # logit = logit_r
        # xout = xout_l


        loss_l1 = at_gen(off_feat, logit_l)
        loss_r1 = conv_gen(feat_x, logit_r)
        loss_distill = loss_l1 + loss_r1

        return logit, xout, loss_distill


if __name__ == '__main__':
    a = torch.randn(4, 200, 9, 9)
    b = torch.randn(4, 2, 9, 9)
    net = DFANet(band=200, classes=16)
    out = net(a, b)
    print(out)

