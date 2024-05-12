from models.TransFuse import *
from models.CoordAttention import CoordAtt
from models.MobileNetV2 import *
from models.PA import PA
from models.basic_modules import *

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


norm_dict = {'BATCH': nn.BatchNorm2d, 'INSTANCE': nn.InstanceNorm2d, 'GROUP': nn.GroupNorm}

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky=True, norm='BATCH', activation=True):
        super().__init__()
        # determine basic attributes
        self.norm_type = norm
        self.activation = activation
        self.leaky = leaky
        padding = (kernel_size - 1) // 2

        # activation, support PReLU and common ReLU
        if self.leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        # instantiate layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm_layer = norm_dict[norm]
        if norm in ['BATCH', 'INSTANCE']:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(8, in_channels)

    def basic_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.act(x)
        return x

    def group_forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        if self.norm_type in ['BATCH', 'INSTANCE']:
            return self.basic_forward(x)
        else:
            return self.group_forward(x)



class PG(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2
        self.conv5_4 = ConvNorm(in_channels[2], in_channels[1], 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], in_channels[1], 3, 1, **kwargs)
        self.conv4_3 = ConvNorm(in_channels[1], in_channels[0], 1, 1, **kwargs)
        self.conv3_0 = ConvNorm(in_channels[0], in_channels[0], 3, 1, **kwargs)
        self.conv_out = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.aff4 = PHFA(channels=96)
        self.aff3 = PHFA(channels=32)


    def forward(self, x3, x4, x5):
        x5_up = self.conv5_4(F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False))
        x4_refine = self.conv4_0(self.aff4(x4, x5_up))
        x4_up = self.conv4_3(F.interpolate(x4_refine, size=x3.shape[2:], mode='bilinear', align_corners=False))
        x3_refine = self.conv3_0(self.aff3(x3, x4_up))
        out = self.conv_out(x3_refine)
        return out



head_list = ['fcn', 'parallel']
head_list = ['fcn', 'parallel']

norm_dict = {'BATCH': nn.BatchNorm2d, 'INSTANCE': nn.InstanceNorm2d, 'GROUP': nn.GroupNorm}


class AttentionConnection(nn.Module):
    def __init__(self, factor=1.0):
        super().__init__()
        self.param = nn.Parameter(torch.Tensor(1).fill_(factor))

    def forward(self, feature, attention):
        return (self.param + attention) * feature


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PHFA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(PHFA, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            # pw
            nn.Conv2d(channels, inter_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, groups=inter_channels, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU6(inplace=True),
            # coordinate attention
            CoordAtt(inter_channels, inter_channels),
            # pw-linear
            nn.Conv2d(inter_channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        xo = self.conv_cat(torch.cat([xo, x, residual], dim=1))
        return xo


class PGPANet(nn.Module):
    def __init__(self, decode_channels=32, dropout=0.1, num_classes=2):
        super(PGPANet, self).__init__()
        self.mbv2 = mobilenet_v2(pretrained=True)
        encoder_channels = [16, 24, 32, 96, 320]#mv2
        self.decode_channels = decode_channels
        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.up16 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.one_stage = PG(in_channels=encoder_channels[2:], out_channels=1)
        self.conv_down = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.skip = AttentionConnection()
        self.head = PA(num_classes, norm_layer=nn.BatchNorm2d, up_kwargs = {'mode': 'bilinear', 'align_corners': True})

        use_deconv = False
        self.up5_4 = ScaleUpsample(use_deconv=use_deconv, num_channels=encoder_channels[-1], scale_factor=2)
        self.up4_3 = ScaleUpsample(use_deconv=use_deconv, num_channels=encoder_channels[-2], scale_factor=2)
        self.up3_2 = ScaleUpsample(use_deconv=use_deconv, num_channels=encoder_channels[-3], scale_factor=2)
        self.up2_1 = ScaleUpsample(use_deconv=use_deconv, num_channels=encoder_channels[-4], scale_factor=2)
        res_unit = ResBottleneck
        # self.exch = ExChange(2, 'ChannelExChange')
        self.conv4_4 = res_unit(encoder_channels[3] + encoder_channels[4], encoder_channels[3])
        self.conv3_1 = res_unit(encoder_channels[2] + encoder_channels[3], encoder_channels[2])
        self.conv2_2 = res_unit(encoder_channels[1] + encoder_channels[2], encoder_channels[1])
        self.conv1_3 = res_unit(encoder_channels[0] + encoder_channels[1], encoder_channels[0])
    def attentionPG(self,attention1,x1_5,x1_4,x1_3,x1_2,x1_1):
        act_attention1 = torch.sigmoid(attention1)
        act_attention11 = self.conv_down(act_attention1)
        attention4_1 = F.interpolate(act_attention11, size=x1_4.shape[2:], mode='bilinear', align_corners=False)
        attention3_1 = F.interpolate(act_attention1, size=x1_3.shape[2:], mode='bilinear', align_corners=False)
        attention2_1 = F.interpolate(act_attention1, size=x1_2.shape[2:], mode='bilinear', align_corners=False)
        attention1_1 = F.interpolate(act_attention1, size=x1_1.shape[2:], mode='bilinear', align_corners=False)

        x1_4 = self.conv4_4(torch.cat([x1_4, self.up5_4(x1_5)], dim=1))
        x1_3 = self.conv3_1(torch.cat([x1_3, self.up4_3(x1_4)], dim=1))
        x1_2 = self.conv2_2(torch.cat([x1_2, self.up3_2(x1_3)], dim=1))
        x1_1 = self.conv1_3(torch.cat([x1_1, self.up2_1(x1_2)], dim=1))

        x1_4 = self.skip(x1_4, attention4_1) + x1_4
        x1_3 = self.skip(x1_3, attention3_1) + x1_3
        x1_2 = self.skip(x1_2, attention2_1) + x1_2
        x1_1 = self.skip(x1_1, attention1_1) + x1_1
        return x1_1, x1_2, x1_3, x1_4


    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.mbv2(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.mbv2(x2)
        attention1 = self.one_stage(x1_3, x1_4, x1_5)
        attention2 = self.one_stage(x2_3, x2_4, x2_5)
        x1_1, x1_2, x1_3, x1_4 = self.attentionPG(attention1, x1_5, x1_4, x1_3, x1_2, x1_1)
        x2_1, x2_2, x2_3, x2_4 = self.attentionPG(attention2, x1_5, x2_4, x2_3, x2_2, x2_1)
        f4 = torch.abs(x1_4 - x2_4)
        f3 = torch.abs(x1_3 - x2_3)
        f2 = torch.abs(x1_2 - x2_2)
        f1 = torch.abs(x1_1 - x2_1)
        features1 = []
        features1.append(f1)
        features1.append(f2)
        features1.append(f3)
        features1.append(f4)
        x1_1, x1_2, x1_3, x1_4 = self.head(*features1)
        x1_3 = self.segmentation_head(self.up4(x1_3))
        x1_3 = torch.sigmoid(x1_3)
        x1_4 = self.segmentation_head(self.up2(x1_4))
        x1_4 = torch.sigmoid(x1_4)
        x1_2 = self.segmentation_head(self.up8(x1_2))
        x1_2 = torch.sigmoid(x1_2)
        x1_1 = self.segmentation_head(self.up16(x1_1))
        x1_1 = torch.sigmoid(x1_1)
        if self.training:
            return x1_4, x1_3, x1_2, x1_1
        else:
            return x1_4


