from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torchvision import models

from nets.layers import FPN, SSH
from nets.mobilenet import MobileNetV1
#from nets.nets import MobileNetV1

class class_head(nn.Module):
    def __init__(self, in_channel, num_anchors):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(in_channel, num_anchors*2, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,3,1).contiguous()

        return x.view(x.shape[0], -1, 2)

class box_head(nn.Module):
    def __init__(self, in_channel, num_anchors):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(in_channel, num_anchors*4, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,3,1).contiguous()

        return x.view(x.shape[0], -1, 4)

class landmark_head(nn.Module):
    def __init__(self, in_channel, num_anchors):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(in_channel, num_anchors*10, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,3,1).contiguous()

        return x.view(x.shape[0], -1, 10)

class RetinaFace(nn.Module):

    def get_class_head(self, fpn_num=3, in_channel=64, num_anchors=2):
        ClassHead = nn.ModuleList()
        for i in range(fpn_num):
            ClassHead.append(class_head(in_channel, num_anchors))
        return ClassHead

    def get_box_head(self, fpn_num=3, in_channel=64, num_anchors=2):
        BoxHead = nn.ModuleList()
        for i in range(fpn_num):
            BoxHead.append(box_head(in_channel, num_anchors))
        return BoxHead

    def get_landmark_head(self, fpn_num=3, in_channel=64, num_anchors=2):
        LandmarkHead = nn.ModuleList()
        for i in range(fpn_num):
            LandmarkHead.append(landmark_head(in_channel, num_anchors))
        return LandmarkHead
    

    def __init__(self, mode = "train"):
        super().__init__()
        self.mode = mode
        # self.backbone = mobilenet()
        
        # 加载预训练好的mobilenet
        backbone = MobileNetV1()
        # checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     name = k[7:]  # remove module.
        #     new_state_dict[name] = v
        # # load params
        # backbone.load_state_dict(new_state_dict)
        return_layers = {'stage1': 1, 'stage2': 2, 'stage3': 3}
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)


        # 三个特征层的通道数
        in_channel_list = [64, 128, 256]
        # 将三个特征层进行FPN，并将输出特征层的通道数都变为64
        self.fpn = FPN(in_channel_list, out_channel=64)
        # 用ssh加强特征
        self.ssh1 = SSH(in_channel=64, out_channel=64)
        self.ssh2 = SSH(in_channel=64, out_channel=64)
        self.ssh3 = SSH(in_channel=64, out_channel=64)
        # 获取方框，与人脸特征点
        self.ClassHead = self.get_class_head(fpn_num=3, in_channel=64, num_anchors=2)
        self.BoxHead = self.get_box_head(fpn_num=3, in_channel=64, num_anchors=2)
        self.LandmarkHead = self.get_landmark_head(fpn_num=3, in_channel=64, num_anchors=2)

    def forward(self, inputs):
        # 利用mobilenet获取三个不同大小的特征层
        # out = self.backbone(inputs)
        out = self.body(inputs)
        out = list(out.values())
        # 利用FPN进行特征融合
        fpn = self.fpn(out)
        # 利用SSH加强特征
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        # 进行边框与人脸特征处理,并堆叠
        classification = [self.ClassHead[i](feature) for i, feature in enumerate(features)]
        box_regression = [self.BoxHead[i](feature) for i, feature in enumerate(features)]
        ldm_regression = [self.LandmarkHead[i](feature) for i, feature in enumerate(features)]
        classification = torch.cat(classification, dim=1)
        box_regression = torch.cat(box_regression, dim=1) 
        ldm_regression = torch.cat(ldm_regression, dim=1)

        # 根据模式来进行返回  
        if self.mode == "train":
            output = (box_regression, classification, ldm_regression)
        else:
            # dim = -1为对最后一维进行softmax
            output = (box_regression, F.softmax(classification, dim=-1), ldm_regression)
        return output

    
