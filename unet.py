import torch
import torch.nn as nn

# إضافة SE-Block (Squeeze-and-Excitation) للتركيز على القنوات اللونية المهمة
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch) # تفعيل الـ SE-Block هنا
        self.relu = nn.ReLU(inplace=True)
        
        # إضافة الـ Residual Connection لتسهيل التدريب ومنع تلاشي التدرج
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.se(out)  # تمرير المخرجات عبر الـ SE-Block
        out += residual     # جمع المدخلات مع المخرجات (Residual)
        return self.relu(out)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512) # ⬅️ إضافة طبقة تشفير رابعة لزيادة العمق
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024) # ⬅️ البوتل-نيك أصبح أعمق (1024 قناة)
        
        # ⬅️ إضافة طبقات فك التشفير المقابلة للعمق الجديد
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3)) # تمرير الطبقة الرابعة
        bn = self.bottleneck(self.pool(c4))
        
        d4 = self.dec4(torch.cat([self.up4(bn), c4], dim=1)) # فك التشفير للطبقة الرابعة
        d3 = self.dec3(torch.cat([self.up3(d4), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        return self.final(d1)