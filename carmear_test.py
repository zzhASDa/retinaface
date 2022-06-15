
from process import Processer
import torch
import numpy as np
import cv2
import time


video_path = 0 # 0表示第一个摄像头，如果为路径则为视频路径
video_fps  = 25.0
video_save_path = ""
image_save_path = "./picture/"


if __name__ == '__main__':

    processer = Processer()
    capture = cv2.VideoCapture(video_path)
    if video_save_path!="":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        
    fps = 0.0
    # cnt = 0
    while(True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # 进行检测
        feature = processer.get_feature(frame)
        frame = processer.draw_feature(frame, feature)
        # frame = np.array(frame)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("video",frame)
        c= cv2.waitKey(1) & 0xff

        # if image_save_path!="":
        #     new_images = processer.cut_image(frame, feature)
        #     for img in new_images:
        #         cv2.imwrite(image_save_path+str(cnt)+".png", img)
        #         cnt += 1

        if video_save_path!="":
            out.write(frame)

        if c==27:
            capture.release()
            break
    print("Video Detection Done!")
    capture.release()
    if video_save_path!="":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()


    