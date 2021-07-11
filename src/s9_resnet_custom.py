import torch.nn as nn
import torch.nn.functional as F

## Customized ResNet model for training CIFAR10
class BasicBlock_Custom(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, make_ind, stride=1):
        super(BasicBlock_Custom, self).__init__()
        self.make_ind = make_ind
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out1 = F.relu(self.shortcut(x))
        if self.make_ind == 'make-block':
            out  = F.relu(self.bn1(self.conv1(out1)))
            out  = F.relu(self.bn2(self.conv2(out)))
            out += out1
        else:
            out  = out1
        return out

class ResNet_Custom(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Custom, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], 'make-block', stride=2)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], 'dont make-block', stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], 'make-block', stride=2)
        self.maxpool4 = nn.MaxPool2d(4, 4)
        self.fc = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def _make_layer(self, block, planes, num_blocks, make_ind, stride):
        strides = [stride-1] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, make_ind, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 32, 32]
        out = self.layer1(out)                 # [B, 128, 16, 16]
        out = self.layer2(out)                 # [B, 256, 8, 8]
        out = self.layer3(out)                 # [B, 512, 4, 4]
        out = self.maxpool4(out)               # [B, 512, 1, 1]
        out = self.fc(out)                     # [B, 10, 1, 1]
        out = out.view(out.size(0), -1)        # [B, 10]
        return F.log_softmax(out, dim=-1)

def ResNet_C():
    return ResNet_Custom(BasicBlock_Custom, [1, 0, 1])