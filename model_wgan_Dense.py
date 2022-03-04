import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LayerNorm([44,44]),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LayerNorm([44,44]),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LayerNorm([22,22]),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
#             nn.LayerNorm([22,22]),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LayerNorm([11,11]),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LayerNorm([11,11]),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LayerNorm([6,6]),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # modification 1: remove sigmoid
        return self.net(x).view(batch_size)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class SingleLayer(nn.Module):
    def __init__(self, inChannels,growthRate):
        super(SingleLayer, self).__init__()
        self.conv =nn.Conv2d(inChannels,growthRate,kernel_size=3,padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

class ResidualDenseBlock(nn.Module):
    def __init__(self, inChannels,growthRate,nDenselayer):
        super(ResidualDenseBlock, self).__init__()
        # self.block= self._make_dense(inChannels,growthRate, nDenselayer)
        self.block = nn.Sequential(*[SingleLayer(inChannels+i*growthRate, growthRate) for i in range(nDenselayer)])
        self.conv = nn.Conv2d(inChannels + growthRate * nDenselayer, growthRate, kernel_size=1)
    # def _make_dense(self,inChannels,growthRate, nDenselayer):
    #     layers = []
    #     for i in range(int(nDenselayer)):
    #         layers.append(SingleLayer(inChannels,growthRate))
    #         inChannels += growthRate
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        out = self.conv(out)
        return out + x  # Single Block Residual (x+block)

class Generator_RDN(nn.Module):
    def __init__(self, scale_factor, inChannels, growthRate, nDenselayer, nBlock):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator_RDN, self).__init__()
        self.nBlock = nBlock
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),  # 3 - 64
            nn.PReLU()
        )
        inChannels = 64
        self.block2 = self._make_block(inChannels, growthRate, nDenselayer,nBlock)  # 64 - 64 + 4*16 - 64
        inChannels = growthRate* nBlock
        # self.block3 = nn.Conv2d(inChannels, 64, kernel_size=3, padding=1)
        self.block3 = nn.Sequential(
            nn.Conv2d(inChannels, 64, kernel_size=1),                               # Global Feature Fusion 64 * 5 - 64
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2)                        # Global Feature Fusion 64 * 5 - 64
        )
        block4 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block4.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block4 = nn.Sequential(*block4)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         xavier(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def _make_block(self, inChannels,growthRate, nDenselayer,nBlock):
        blocks =[]
        for i in range(int(nBlock)):
            blocks.append(ResidualDenseBlock(inChannels,growthRate,nDenselayer))
            # inChannels += growthRate* nDenselayer
        return nn.ModuleList(blocks)

    def forward(self, x):
        block1 = self.block1(x)                                        # 3 - 64
        block2 = block1
        local_features = []
        for i in range(self.nBlock):
            block2 = self.block2[i](block2)                           # 64 - 64 + 4*16 - 64
            local_features.append(block2)                             # Global Feature Fusion 64 * 5 - 64
        block3 = self.block3(torch.cat(local_features, 1)) + block1   # Residual 64 + 64
        block4 = self.block4(block3)                                  #Upscale 64 - 3

        return (torch.tanh(block4) + 1) / 2