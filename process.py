import numpy as np
import cv2
import torch
from torchvision.ops import nms
from nets.retinaface import RetinaFace
from utils.prior_box import PriorBox
from utils.box_utils import decode, decode_landm# , nms

class Processer(object):
    def __init__(self):
        # 加载模型
        self.model_path = "./weights/mobilenet0.25_epoch_10net.pth"
        self.net = RetinaFace(mode="test")
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        
        self.image_size = (640, 640)
        self.priors = PriorBox(image_size=self.image_size).forward()
        self.rgb_mean = (104, 117, 123)
        self.cuda = True
        self.variances = [0.1, 0.2]
        self.confidence_threshold = 0.9
        self.nms_threshold = 0.2
        self.keep_top_k = 750
        
        if self.cuda:
            self.net.cuda()
    # net = load_model(net, args.trained_model, args.cpu)

    # 对图片进行大小的变换，用灰色填充边框
    def letterbox_image(self, image, size):
        ih, iw, _   = np.shape(image)
        h, w        = size
        scale       = min(w/iw, h/ih)
        nw          = int(iw*scale)
        nh          = int(ih*scale)

        image       = cv2.resize(image, (nw, nh))
        new_image   = np.ones([size[1], size[0], 3]) * 128
        new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
        return new_image


    # 用于将变形后的图片还原回原图
    def retinaface_correct_boxes(self, result, input_shape, image_shape):
        new_shape   = image_shape*np.min(input_shape/image_shape)

        offset      = (input_shape - new_shape) / 2. / input_shape
        scale       = input_shape / new_shape
        
        scale_for_boxs      = [scale[1], scale[0], scale[1], scale[0]]
        scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0]]

        offset_for_boxs         = [offset[1], offset[0], offset[1],offset[0]]
        offset_for_landmarks    = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0]]

        result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
        result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

        return result
    

    def __call__(self, image):
        old_image = image.copy()
        image = np.array(image, np.float32)
        img_h, img_w, _ = np.shape(image)

        # 用于将边框与人脸关键点扩张回原图大小
        scale = [
            img_w, img_h, img_w, img_h
        ]
        scale_for_landmarks = [
            img_w, img_h, img_w, img_h,
            img_w, img_h, img_w, img_h,
            img_w, img_h
        ]

        # 将图像转换成指定大小
        image = self.letterbox_image(image, [self.image_size[0], self.image_size[1]])
        
        with torch.no_grad():
            # 训练时加上了rgb_mean，处理时需要减掉
            image = image - np.array(self.rgb_mean, np.float32)
            # 进行维度的变换，将三个通道提到第一维
            image = image.transpose(2, 0, 1)
            # 扩展batch_size维度
            image = torch.from_numpy(image).unsqueeze(0).type(torch.FloatTensor)

            if self.cuda:
                self.priors = self.priors.cuda()
                image = image.cuda()

            loc, conf, landms = self.net(image)
            
            # 对预测框进行解码，由于batch_size为1，所以将该维度去除
            boxes   = decode(loc.data.squeeze(0), self.priors, self.variances)
            # 获取置信度
            conf    = conf.data.squeeze(0)[:, 1:2]
            # 对人脸关键点进行解码，由于batch_size为1，所以将该维度去除
            landms  = decode_landm(landms.data.squeeze(0), self.priors, self.variances)

            # 将结果堆叠起来
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            # 去除小于置信度阈值的
            mask = boxes_conf_landms[:, 4] >= self.confidence_threshold
            boxes_conf_landms = boxes_conf_landms[mask]
            # print(boxes_conf_landms.shape)
            # 进行非极大值抑制
            keep = nms(boxes_conf_landms[:,:4], boxes_conf_landms[:,4], self.nms_threshold)
            print(keep)
            boxes_conf_landms = boxes_conf_landms[keep]
            boxes_conf_landms = boxes_conf_landms.cpu().numpy()

            if len(boxes_conf_landms) <= 0:
                return old_image

            #
            #print(boxes_conf_landms.shape)
            boxes_conf_landms = self.retinaface_correct_boxes(boxes_conf_landms, \
                np.array([self.image_size[0], self.image_size[1]]), np.array([img_h, img_w]))
            
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            #---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            #---------------------------------------------------#
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            #print(b[0], b[1], b[2], b[3], b[4])
            #---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            #---------------------------------------------------#
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
        return old_image