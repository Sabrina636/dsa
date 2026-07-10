import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义基础卷积块
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),# 批标准化，加速训练并提高稳定性
            nn.ReLU(inplace=True),# ReLU激活函数，inplace=True减少内存消耗
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            #尺寸不变
        )

    def forward(self, x):
        x = self.conv(x)# 应用卷积序列
        return x

# 定义上采样模块
# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),# 双线性上采样（默认），将特征图尺寸扩大2倍
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.up(x)
#         return x


#改进1：用卷积上采样
class up_conv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, ch_in, ch_out, x2_ch, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv_block(ch_in + x2_ch, ch_out)
        else:
            self.up = nn.ConvTranspose2d(ch_in,ch_out // 2, kernel_size=2, stride=2)
            self.conv = conv_block(ch_in + x2_ch, ch_out)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 改进2：注意力机制
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_att(x)
        x = x * channel_att

        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * spatial_att


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # 注意力模块
        self.cbam1 = CBAM(512)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(64)

        # 解码器（调整初始化参数）
        self.Up5 = up_conv(ch_in=1024, ch_out=512, x2_ch=512)  # x4的通道是512
        self.Up4 = up_conv(ch_in=512, ch_out=256, x2_ch=256)  # x3的通道是256
        self.Up3 = up_conv(ch_in=256, ch_out=128, x2_ch=128)  # x2的通道是128
        self.Up2 = up_conv(ch_in=128, ch_out=64, x2_ch=64)  # x1的通道是64

        # 最终卷积层
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):

        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # 解码部分
        x4 = self.cbam1(x4)  # 应用注意力到跳跃连接特征
        d5 = self.Up5(x5, x4)  # 传入x5和x4

        x3 = self.cbam2(x3)
        d4 = self.Up4(d5, x3)

        x2 = self.cbam3(x2)
        d3 = self.Up3(d4, x2)

        x1 = self.cbam4(x1)
        d2 = self.Up2(d3, x1)

        # 最终输出
        d1 = self.Conv_1x1(d2)
        return d1
