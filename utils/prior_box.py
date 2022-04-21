import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        # 每个特征图中边框的初始大小，一个像素点对应2个边框
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        # 每个特征图相较于原图缩小的倍数
        self.steps = [8, 16, 32]
        # 输入图片的大小
        self.image_size = image_size
        # 特征图的大小
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1] # 特征框边长相较于原图的比例
                    s_ky = min_size / self.image_size[0]
                    x = j + 0.5 # 像素中心点坐标（比如（0,0）中心点即为0=（0.5,0.5））
                    y = i + 0.5
                    cx = x / f[1] # 中心点除以特征图大小，归一化
                    cy = y / f[0]
                    anchors += [cx, cy, s_kx, s_ky]
                    # dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    # dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    # for cy, cx in product(dense_cy, dense_cx):
                    #     print(dense_cx[0], cx, dense_cy[0], cy)
                    #     anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        # if self.clip:
        #     output.clamp_(max=1, min=0)
        return output

if __name__ == "__main__":
    priorbox = PriorBox(image_size=(600,600))
    box = priorbox.forward()
    print(box.shape)

