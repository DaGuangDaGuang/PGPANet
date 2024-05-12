from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate


class PA(nn.Module):
    def __init__(self, norm_layer=None, fpn_inchannels=[16, 24, 32, 96],
                 fpn_dim=64, up_kwargs=None):
        super(PA, self).__init__()
        assert up_kwargs is not None
        self._up_kwargs = up_kwargs
        fpn_lateral = []
        for fpn_inchannel in fpn_inchannels[:-1]:
            fpn_lateral.append(nn.Sequential(
                nn.Conv2d(fpn_inchannel, fpn_dim, kernel_size=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_lateral = nn.ModuleList(fpn_lateral)
        fpn_out = []
        for _ in range(len(fpn_inchannels) - 1):
            fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_out = nn.ModuleList(fpn_out)
        self.e5conv = nn.Sequential(nn.Conv2d(fpn_inchannels[-1], fpn_dim, 3, padding=1, bias=False),
                                    norm_layer(fpn_dim),
                                    nn.ReLU())
        self.gidp_d4 = GIDP(fpn_dim, fpn_dim, fpn_dim // 2)
        self.gidp_d3 = GIDP(fpn_dim, fpn_dim, fpn_dim // 2)
        self.gidp_d2 = GIDP(fpn_dim, fpn_dim, fpn_dim // 2)

        self.cfb_d4 = nn.ModuleList(
            [Fusion(fpn_dim, fpn_dim // 2, norm_layer=norm_layer),
             Fusion(fpn_dim, fpn_dim // 2, norm_layer=norm_layer),
             Fusion(fpn_dim, fpn_dim // 2, norm_layer=norm_layer)]
        )
        self.cfb_d3 = nn.ModuleList(
            [Fusion(fpn_dim, fpn_dim // 2, norm_layer=norm_layer),
             Fusion(fpn_dim, fpn_dim // 2, norm_layer=norm_layer),
             Identity()]
        )
        self.cfb_d2 = nn.ModuleList(
            [Fusion(fpn_dim, fpn_dim // 2, norm_layer=norm_layer),
             Identity(),
             Identity()]
        )
    def forward(self, *inputs):
        e5 = inputs[-1]
        feat = self.e5conv(e5)
        fpn_features = [feat]
        descriptors = [0, 0, 0]
        for i in reversed(range(len(inputs) - 1)):
            feat_i = self.fpn_lateral[i](inputs[i])
            feat_up = interpolate(feat, feat_i.size()[2:], **self._up_kwargs)
            if i == 2:
                descriptors[i] = self.gidp_d4(feat)
            if i == 1:
                descriptors[i] = self.sab_d3(feat)
            if i == 0:
                descriptors[i] = self.gidp_d2(feat)
            feat_up = self.cfb_d4[i](feat_i, feat_up, descriptors[2])
            feat_up = self.cfb_d3[i](feat_i, feat_up, descriptors[1])
            feat = self.cfb_d2[i](feat_i, feat_up, descriptors[0])
            fpn_features.append(feat)
        return fpn_features


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input_de):
        return input_de

class GIDP(nn.Module):
    def __init__(self, c_in, c_feat, top_k):
        super(GIDP, self).__init__()
        self.c_feat = c_feat
        self.topk = top_k
        self.conv_feat = nn.Conv2d(c_in, c_feat, kernel_size=1)

    def forward(self, input: torch.Tensor):

        b, c, h, w = input.size()
        feat = self.conv_feat(input).view(b, self.c_feat, -1)  # feature map
        atten = input.view(b, c, -1)  # attention map
        atten = torch.topk(atten, self.topk, dim=1)[0]
        atten = F.softmax(atten, dim=-1)
        descriptors = torch.bmm(feat, atten.permute(0, 2, 1))  # (c_feat, c_atten)
        return descriptors

class SAAA(nn.Module):

     def __init__(self, top_k, c_de):
         super(SAAA, self).__init__()
         self.topk = top_k
         self.out_conv = nn.Conv2d(c_de, c_de, kernel_size=1)

     def forward(self, descriptors: torch.Tensor, input_de: torch.Tensor):
         b, c, h, w = input_de.size()
         atten_vectors = F.softmax(torch.topk(input_de, self.topk, dim=1)[0])
         output = descriptors.matmul(atten_vectors.view(b, self.topk, -1)).view(b, -1, h, w)
         return self.out_conv(output)

class Out(nn.Module):

    def __init__(self, c_en, c_de):
        super(Out, self).__init__()
        self.c_en = c_en
        self.c_de = c_de

    def forward(self, input_en: torch.Tensor, input_de: torch.Tensor, gate_map):
        b, c, h, w = input_de.size()
        input_en = input_en.view(b, self.c_en, -1)
        energy = input_de.view(b, self.c_de, -1).matmul(input_en.transpose(-1, -2))
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        channel_attention_map = torch.softmax(energy_new, dim=-1)
        input_en = channel_attention_map.matmul(input_en).view(b, -1, h, w)  # channel_attention_feat
        gate_map = torch.sigmoid(gate_map)
        input_en = input_en.mul(gate_map)
        return input_en


class Fusion(nn.Module):

    def __init__(self, fpn_dim=256, c_atten=256, norm_layer=None, ):
        super(Fusion, self).__init__()
        self.saaa = SAAA(c_atten, fpn_dim)
        self.out = Out(fpn_dim, fpn_dim)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            norm_layer(fpn_dim),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, input_en: torch.Tensor, input_de: torch.Tensor, global_descripitors: torch.Tensor):
        feat_global = self.saaa(global_descripitors, input_de)
        feat_local = self.gamma * self.out(input_en, input_de, feat_global) + input_en
        return self.conv_fusion(input_de + self.alpha * feat_global + self.beta * feat_local)

