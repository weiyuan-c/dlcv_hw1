import torch.nn as nn
import torchvision.models
from torchvision.models import ResNet50_Weights
from torchvision.models import VGG16_BN_Weights
from torchvision.models import VGG16_Weights

# Classifer

# model A
def conv_layer(input_size, output_size):
    return nn.Sequential(nn.Conv2d(input_size, output_size, 3, 1, 1),
                         nn.BatchNorm2d(output_size),
                         nn.ReLU(),
                         nn.MaxPool2d(2, 2, 0))
    
def residual(input_size):
    return nn.Sequential(nn.Conv2d(input_size, input_size, 3, 1, 1),
                         nn.BatchNorm2d(input_size),
                         nn.ReLU())

class modelA(nn.Module):
    def __init__(self):
        super(modelA, self).__init__()
        self.l1 = conv_layer(3, 64) # [64, 64, 64]
        self.r1 = residual(64)
        self.r2 = residual(64)
        self.l2 = conv_layer(64, 64) # [64, 32, 32]
        self.l3 = conv_layer(64, 128) # [128, 16, 16]
        self.r3 = residual(128)
        self.r4 = residual(128)
        self.l4 = conv_layer(128, 128) # [256, 8, 8]
        self.l5 = conv_layer(128, 256) # [256, 4, 4]
        self.r5 = residual(256)
        self.r6 = residual(256)
        self.l6 = conv_layer(256, 512) # [128, 2, 2]
        self.fc = nn.Linear(512*2*2, 50)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x1 = self.r2(self.r1(x))
        x = self.relu(x + x1)
        x = self.l3(self.l2(x))
        x1 = self.r4(self.r3(x))
        x = self.relu(x + x1)
        x = self.l5(self.l4(x))
        x1 = self.r6(self.r5(x))
        x = self.relu(x + x1)
        x = self.l6(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

# model B
class Classifier(nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        super(Classifier, self).__init__()
        self.cnn = torchvision.models.resnet50(weights=weights)
        self.fc = nn.Linear(1000, 50)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out

# Semantic segmentation

# VGG16-FCN32
class vgg_fcn32(nn.Module):
    def __init__(self, n_class=7):
        super(vgg_fcn32, self).__init__()
        self.vgg = torchvision.models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        self.fc1 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        self.conv = nn.Conv2d(4096, n_class, 1)
        self.up_sample = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False, padding=16)

    def forward(self, x):
        x = self.vgg(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.conv(x)
        x = self.up_sample(x)

        return  x

# VGG16-FCN-1
class vgg_fcn8(nn.Module):
    def __init__(self, n_class=7, weights=VGG16_Weights.DEFAULT):
        super(vgg_fcn8, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=weights).features

        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        self.conv1 = nn.Conv2d(4096, n_class, 1)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(n_class, n_class, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(512, n_class, 1)
        self.up2 = self.up1 = nn.Sequential(
            nn.ConvTranspose2d(n_class, n_class, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(256, n_class, 1)
        self.up3 = nn.ConvTranspose2d(n_class, n_class, 16, 8, 4)
    def forward(self, x):
        x8 = self.vgg[:17](x) # pool 3 [, 256, 64, 64]
        x16 = self.vgg[17:24](x8) # pool4 [, 512, 32, 32]
        x32 = self.vgg[24:](x16) # pool 5 [, 512, 16, 16]

        x = self.fc6(x32) # [, 4096, 16, 16]
        x = self.fc7(x) # [, 4096, 16, 16]
        x = self.conv1(x) # [, 7, 16, 16]
        x = self.up1(x) # [, 7, 32, 32]
        x1 = self.conv2(x16) # [, 7, 32, 32]
        x = self.up2(x + x1) # [, 7, 64, 64]
        x2 = self.conv3(x8) # [, 7, 64, 64]
        x = x + x2

        return self.up3(x)


# VGG16-FCN8-2
class vgg_fcn8_2(nn.Module):
    def __init__(self, n_class=7, weights=VGG16_Weights.DEFAULT):
        super(vgg_fcn8_2, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=weights).features

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.bn1 = nn.BatchNorm2d(512)

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.bn2 = nn.BatchNorm2d(256)

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        self.classifier = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        x8 = self.vgg[:17](x)
        x16 = self.vgg[17:24](x8)
        x32 = self.vgg[24:](x16)

        x = self.layer1(x32)
        x = self.bn1(x + x16)
        x = self.layer2(x)
        x = self.bn2(x + x8)
        x = self.layer5(self.layer4(self.layer3(x)))

        return self.classifier(x)
