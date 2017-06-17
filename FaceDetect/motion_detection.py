# -*- coding: utf-8 -*-
import cv2
import imutils
import time
ISOTIMEFORMAT='%Y-%m-%d %X'


def have_person(record1):
    method1 = 0
    method2 = 0
    for item in record1:
        if item is None:
            continue
        if item[0] > 0:
            method1 = method1+1
        if item[0] > 1:
            method2 = method2+1
    if method2 > 2 or method1 > 3:
        return True
    return False

video_path = "picture/video.mp4"
choose = 1
if choose == 0:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(video_path)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.createBackgroundSubtractorKNN()
record = [None, None, None, None, None]
state = False
current = 0
last_time = time.strftime(ISOTIMEFORMAT, time.localtime())
while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        print('end:' + time.strftime(ISOTIMEFORMAT, time.localtime()))
        break
    frame = imutils.resize(frame, width=500)

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)
    # fgmask1 = fgbg1.apply(frame)
    # fgmask1 = cv2.morphologyEx(fgmask1, cv2.MORPH_OPEN, kernel)
    # fgmask1 = cv2.dilate(fgmask1, None, iterations=2)
    img, cnts, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 运动区域计数
    nums = 0
    # 遍历轮廓
    for c in cnts:
        # if the contour is too small, ignore it
        perimeter = cv2.arcLength(c, True)
        if perimeter < 60:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        # 计算轮廓的边界框，在当前帧中画出该框
        nums = nums + 1
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(fgmask, (x, y), (x+w, y+h), 255, 2)

    # 记录信息
    record[current] = (nums, time.strftime(ISOTIMEFORMAT, time.localtime()))
    current = (current+1) % 5
    temp_state = have_person(record)
    if state != temp_state and record[0][1] != last_time:
        state = temp_state
        last_time = record[0][1]
        if state:
            print ("有人活动时段"+record[0][1])

        else:
            print ("有人活动时段结束"+record[0][1])
    # print("motion region: %d" % (nums))
    cv2.imshow('frame', fgmask)
    # cv2.imshow('frame1', fgmask1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
