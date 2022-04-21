import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, outp, stride):
    return nn.Sequential(
        nn.Conv2d(inp, outp, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outp),
    )

def conv_bn_ReLU(inp, outp, stride):
    return nn.Sequential(
        nn.Conv2d(inp, outp, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU6(inplace=True)
    )
def conv_bn1x1(inp, outp, stride):
    return nn.Sequential(
        nn.Conv2d(inp, outp, 1, stride, 0, bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU6(inplace=True)
    )

# 将三个特征层进行融合， 并进行通道数变换
class FPN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super().__init__()
        self.output1 = conv_bn1x1(in_channel_list[0], out_channel, stride = 1)
        self.output2 = conv_bn1x1(in_channel_list[1], out_channel, stride = 1)
        self.output3 = conv_bn1x1(in_channel_list[2], out_channel, stride = 1)

        self.merge1 = conv_bn_ReLU(out_channel, out_channel, 1)
        self.merge2 = conv_bn_ReLU(out_channel, out_channel, 1)

    def forward(self, inputs):

        # output1到3，特征图大小为从大到小
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        up3 = F.interpolate(output3, size=output2.shape[2:], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge1(output2)

        up2 = F.interpolate(output2, size=output1.shape[2:], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge2(output1)

        return (output1, output2, output3)

# 对每个特征层进行多尺度加强感受野，通道数变换
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # 保证out_channel能被四整除
        assert out_channel % 4 == 0
        
        self.conv3x3 = conv_bn_ReLU(in_channel, out_channel//2, 1)

        self.conv5x5_1 = conv_bn(in_channel, out_channel//4, 1)
        self.conv5x5_2 = conv_bn_ReLU(out_channel//4, out_channel//4, 1)

        self.conv7x7_2 = conv_bn(out_channel//4, out_channel//4, 1)
        self.conv7x7_3 = conv_bn_ReLU(out_channel//4, out_channel//4, 1)

    def forward(self, x):
        out1 = self.conv3x3(x)

        out2_1 = self.conv5x5_1(x)
        out2 = self.conv5x5_2(out2_1)

        out3_1 = self.conv7x7_2(out2_1)
        out3 = self.conv7x7_3(out3_1)

        out = torch.cat([out1, out2, out3], dim=1)
        out = F.relu(out)
        return out