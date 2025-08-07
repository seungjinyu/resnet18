import torch.nn as nn

class ResNetStem(nn.Module):
    def __init__(self):

        super().__init__()
        self.in_channels = 64 

        # Stem
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Resnet basic blocks
        self.layer1 = self._make_layer(ResNetBasicBlock,64,2)
        self.layer2 = self._make_layer(ResNetBasicBlock,128,2,stride=2)
        self.layer3 = self._make_layer(ResNetBasicBlock,256,2,stride=2)
        self.layer4 = self._make_layer(ResNetBasicBlock,512,2,stride=2)

    def _make_layer(self, block,out_channels, blocks,stride=1):
        # downsample 
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers=[]
        layers.append(block(self.in_channels, out_channels,stride,downsample))
        self.in_channels = out_channels

        for _ in range(1,blocks):
            layers.append(block(out_channels,out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        print(f"Input: {x.shape}")
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f"After maxpool: {x.shape}")

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNetBasicBlock(nn.Module):


    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1 , downsample=None):
        super().__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # conv2
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3, stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
