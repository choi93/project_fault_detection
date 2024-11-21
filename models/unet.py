import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    U-Net 신경망 구조
    - 이미지 세그멘테이션을 위한 인코더-디코더 구조
    - 수축경로(Contracting path)와 확장경로(Expansive path)로 구성
    """
    def __init__(self):
        super(UNet, self).__init__()
        # 수축 경로 (인코더)
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.1)

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.1)

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.2)

        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(0.2)

        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.drop5 = nn.Dropout(0.3)

        # Expansive path
        self.upconv6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.conv6_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.drop6 = nn.Dropout(0.2)

        self.upconv7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.conv7_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.drop7 = nn.Dropout(0.2)

        self.upconv8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.conv8_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.drop8 = nn.Dropout(0.1)

        self.upconv9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.conv9_1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.drop9 = nn.Dropout(0.1)

        self.out = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Contracting path
        c1 = F.elu(self.conv1_1(x))
        c1 = self.drop1(c1)
        c1 = F.elu(self.conv1_2(c1))
        p1 = self.pool1(c1)

        c2 = F.elu(self.conv2_1(p1))
        c2 = self.drop2(c2)
        c2 = F.elu(self.conv2_2(c2))
        p2 = self.pool2(c2)

        c3 = F.elu(self.conv3_1(p2))
        c3 = self.drop3(c3)
        c3 = F.elu(self.conv3_2(c3))
        p3 = self.pool3(c3)

        c4 = F.elu(self.conv4_1(p3))
        c4 = self.drop4(c4)
        c4 = F.elu(self.conv4_2(c4))
        p4 = self.pool4(c4)

        c5 = F.elu(self.conv5_1(p4))
        c5 = self.drop5(c5)
        c5 = F.elu(self.conv5_2(c5))

        # Expansive path
        u6 = self.upconv6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = F.elu(self.conv6_1(u6))
        c6 = self.drop6(c6)
        c6 = F.elu(self.conv6_2(c6))

        u7 = self.upconv7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = F.elu(self.conv7_1(u7))
        c7 = self.drop7(c7)
        c7 = F.elu(self.conv7_2(c7))

        u8 = self.upconv8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = F.elu(self.conv8_1(u8))
        c8 = self.drop8(c8)
        c8 = F.elu(self.conv8_2(c8))

        u9 = self.upconv9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = F.elu(self.conv9_1(u9))
        c9 = self.drop9(c9)
        c9 = F.elu(self.conv9_2(c9))

        outputs = torch.sigmoid(self.out(c9))
        return outputs 