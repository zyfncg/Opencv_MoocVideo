#encoding=utf-8
from __future__ import print_function

import os

import numpy
from PIL import Image
import cv2

from image_handle import projectTransform
from image_handle import multiOperation
# from image_handle import bothEnhance
import imutils
# mouse_params = {'window_name': 'video', 'draw': False, 'start': None, 'end': None,'image': None, 'draw_finish': False}
# def on_mouse(event, x, y, flags, param):
#
#         if event == cv2.EVENT_LBUTTONDOWN:
#             param['draw'] = True
#             param['start'] = (x, y)
#         elif event == cv2.EVENT_LBUTTONUP:
#             param['draw'] = False
#             param['end'] = (x, y)
#             image_show = param['image'].copy()
#             cv2.rectangle(image_show, param['start'], (x, y), (0, 255, 255), 2)
#             cv2.imshow(mouse_params['window_name'], image_show)
#         elif event == cv2.EVENT_MOUSEMOVE:
#             if param['draw']:
#                 image_show = param['image'].copy()
#                 cv2.rectangle(image_show, param['start'], (x, y), (0, 255, 255), 2)
#                 cv2.imshow(mouse_params['window_name'], image_show)


mouse_params = {'window_name': 'video', 'state': 0, 'coordinate': [None, None, None, None], 'markFinish': False}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        param['coordinate'][param['state']] = (x, y)
        param['state'] = param['state'] + 1
        if param['state'] == 4:
            param['state'] = 0

        image_show = param['image']
        cv2.circle(image_show, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow(mouse_params['window_name'], image_show)


video_path = "../file/test61.mp4"

choose = 1
if choose == 0:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(video_path)

(grabbed, frame) = camera.read()
if grabbed:
    # frame = imutils.resize(frame, width=300)
    cv2.imshow(mouse_params['window_name'], frame)
    mouse_params['image'] = frame
    cv2.setMouseCallback(mouse_params['window_name'], on_mouse, mouse_params)
    cv2.waitKey(0)

# 删去最开始标记的图片框
cv2.destroyWindow(mouse_params['window_name'])

# 获取视频参数
fps = camera.get(cv2.CAP_PROP_FPS)
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('result.avi', fourcc, 20.0, size, True)

while True:
    # print("hello")
    (grabbed, frame) = camera.read()
    if choose == 1 and not grabbed:
        break
    # frame = imutils.resize(frame, width=300)

    frameClone = frame.copy()
    for i in range(4):
        cv2.circle(frameClone, mouse_params['coordinate'][i], 2, (0, 255, 0), -1)

    cv2.imshow("original", frameClone)
    print(mouse_params['coordinate'])
    # 投影并展示
    handled_image=projectTransform(frameClone,mouse_params['coordinate'])
    # cv2.imshow("handled_image",handled_image)
    # 转成PIL.Image格式进行增强操作
    cv2_im = cv2.cvtColor(handled_image,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    enhanced_image=multiOperation(pil_im)

    # 转成cv2.Image格式进行增强
    pil_image = enhanced_image.convert('RGB')
    open_cv_image = numpy.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    contrast = numpy.column_stack((handled_image, open_cv_image))

    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_RGB2BGR)
    videoWriter.write(contrast)
    # cv2.imshow("enhanced_image", open_cv_image)
    cv2.imshow("enhanced_image", contrast)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
videoWriter.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
