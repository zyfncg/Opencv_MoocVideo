from __future__ import print_function
import imutils
import cv2

mouse_params = {'window_name': 'video', 'draw': False, 'start': None, 'end': None, 'image': None, 'draw_finish': False}


def on_mouse(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            param['draw'] = True
            param['start'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            param['draw'] = False
            param['end'] = (x, y)
            image_show = param['image'].copy()
            cv2.rectangle(image_show, param['start'], (x, y), (0, 255, 255), 2)
            cv2.imshow(mouse_params['window_name'], image_show)
        elif event == cv2.EVENT_MOUSEMOVE:
            if param['draw']:
                image_show = param['image'].copy()
                cv2.rectangle(image_show, param['start'], (x, y), (0, 255, 255), 2)
                cv2.imshow(mouse_params['window_name'], image_show)


video_path = "picture/test2.flv"
choose = 1
if choose == 0:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(video_path)

(grabbed, frame) = camera.read()
if grabbed:
    frame = imutils.resize(frame, width=300)
    cv2.imshow(mouse_params['window_name'], frame)
    mouse_params['image'] = frame
    cv2.setMouseCallback(mouse_params['window_name'], on_mouse, mouse_params)
    cv2.waitKey(0)

while True:
    (grabbed, frame) = camera.read()
    if choose == 1 and not grabbed:
        break
    frame = imutils.resize(frame, width=300)

    frameClone = frame.copy()
    cv2.rectangle(frameClone, mouse_params['start'], mouse_params['end'], (0, 255, 255), 2)
    cv2.imshow("result", frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
