import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    """两次 Conv-BN-ReLU，尺寸不变"""
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def upsample_and_pad(x1, x2):
    """将 x1 上采样 2× 后 pad 到与 x2 相同的 H×W"""
    x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
    diffY = x2.size(2) - x1.size(2)
    diffX = x2.size(3) - x1.size(3)
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                     diffY // 2, diffY - diffY // 2])
    return x1


class UNetPlusPlus(nn.Module):

    def __init__(self, img_ch=3, output_ch=1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        nb = [64, 128, 256, 512, 1024]   # 各层通道数

        # ── 编码器 ──────────────────────────────
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc0  = conv_block(img_ch,   nb[0])   # X_{0,0}
        self.enc1  = conv_block(nb[0],    nb[1])   # X_{1,0}
        self.enc2  = conv_block(nb[1],    nb[2])   # X_{2,0}
        self.enc3  = conv_block(nb[2],    nb[3])   # X_{3,0}
        self.enc4  = conv_block(nb[3],    nb[4])   # X_{4,0} 瓶颈

        # ── 嵌套节点 ────────────────────────────
        # 输入通道 = 同层前驱节点通道之和 + 上层上采样通道
        # 第 1 列（j=1）
        self.node_3_1 = conv_block(nb[3] * 1 + nb[4], nb[3])
        self.node_2_1 = conv_block(nb[2] * 1 + nb[3], nb[2])
        self.node_1_1 = conv_block(nb[1] * 1 + nb[2], nb[1])
        self.node_0_1 = conv_block(nb[0] * 1 + nb[1], nb[0])

        # 第 2 列（j=2）
        self.node_2_2 = conv_block(nb[2] * 2 + nb[3], nb[2])
        self.node_1_2 = conv_block(nb[1] * 2 + nb[2], nb[1])
        self.node_0_2 = conv_block(nb[0] * 2 + nb[1], nb[0])

        # 第 3 列（j=3）
        self.node_1_3 = conv_block(nb[1] * 3 + nb[2], nb[1])
        self.node_0_3 = conv_block(nb[0] * 3 + nb[1], nb[0])

        # 第 4 列（j=4）
        self.node_0_4 = conv_block(nb[0] * 4 + nb[1], nb[0])

        # ── 深监督输出头 ─────────────────────────
        self.out1 = nn.Conv2d(nb[0], output_ch, kernel_size=1)
        self.out2 = nn.Conv2d(nb[0], output_ch, kernel_size=1)
        self.out3 = nn.Conv2d(nb[0], output_ch, kernel_size=1)
        self.out4 = nn.Conv2d(nb[0], output_ch, kernel_size=1)

    def forward(self, x):
        # ── 编码器 ──────────────────────────────
        x0_0 = self.enc0(x)                       # [B,  64, H,    W   ]
        x1_0 = self.enc1(self.pool(x0_0))         # [B, 128, H/2,  W/2 ]
        x2_0 = self.enc2(self.pool(x1_0))         # [B, 256, H/4,  W/4 ]
        x3_0 = self.enc3(self.pool(x2_0))         # [B, 512, H/8,  W/8 ]
        x4_0 = self.enc4(self.pool(x3_0))         # [B,1024, H/16, W/16]

        # ── 第 1 列（j=1）───────────────────────
        x3_1 = self.node_3_1(torch.cat([x3_0, upsample_and_pad(x4_0, x3_0)], dim=1))
        x2_1 = self.node_2_1(torch.cat([x2_0, upsample_and_pad(x3_1, x2_0)], dim=1))
        x1_1 = self.node_1_1(torch.cat([x1_0, upsample_and_pad(x2_1, x1_0)], dim=1))
        x0_1 = self.node_0_1(torch.cat([x0_0, upsample_and_pad(x1_1, x0_0)], dim=1))

        # ── 第 2 列（j=2）───────────────────────
        x2_2 = self.node_2_2(torch.cat([x2_0, x2_1, upsample_and_pad(x3_1, x2_0)], dim=1))
        x1_2 = self.node_1_2(torch.cat([x1_0, x1_1, upsample_and_pad(x2_2, x1_0)], dim=1))
        x0_2 = self.node_0_2(torch.cat([x0_0, x0_1, upsample_and_pad(x1_2, x0_0)], dim=1))

        # ── 第 3 列（j=3）───────────────────────
        x1_3 = self.node_1_3(torch.cat([x1_0, x1_1, x1_2, upsample_and_pad(x2_2, x1_0)], dim=1))
        x0_3 = self.node_0_3(torch.cat([x0_0, x0_1, x0_2, upsample_and_pad(x1_3, x0_0)], dim=1))

        # ── 第 4 列（j=4）───────────────────────
        x0_4 = self.node_0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, upsample_and_pad(x1_3, x0_0)], dim=1))

        # ── 输出 ────────────────────────────────
        if self.deep_supervision and self.training:
            # 训练：四个监督头平均
            return (self.out1(x0_1) + self.out2(x0_2) +
                    self.out3(x0_3) + self.out4(x0_4)) / 4
        else:
            # 推理：只用最深路径
            return self.out4(x0_4)


