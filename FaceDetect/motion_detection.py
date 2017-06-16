# -*- coding: utf-8 -*-
import cv2

video_path = "picture/video.mp4"
choose = 1
if choose == 0:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(video_path)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg1 = cv2.createBackgroundSubtractorKNN()

while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        print('end')
        break

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask1 = fgbg1.apply(frame)
    fgmask1 = cv2.morphologyEx(fgmask1, cv2.MORPH_OPEN, kernel)
    img, cnts, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓
    for c in cnts:
        # if the contour is too small, ignore it
        perimeter = cv2.arcLength(c, True)
        if perimeter < 20:
            continue
        print('detect')
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        # 计算轮廓的边界框，在当前帧中画出该框
        (x, y, w, h) = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)
        fgmask = cv2.add(fgmask, rect)

    cv2.imshow('frame', fgmask)
    cv2.imshow('frame1', fgmask1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
