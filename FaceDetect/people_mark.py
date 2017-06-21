from __future__ import print_function
import imutils
from facedetector import FaceDetector
import cv2

video_path = "picture/video.mp4"
choose = 1
if choose == 0:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(video_path)

fd = FaceDetector("haarcascade_frontalface_default.xml")
pfd = FaceDetector("haarcascade_profileface.xml")
fd = FaceDetector("haarcascade_upperbody.xml")
while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=6, minSize=(25, 25))
    # pfaceRects = pfd.detect(gray, scaleFactor=1.1, minNeighbors=6, minSize=(19, 19))
    frameClone = frame.copy()

    for (fX, fY, fW, fH) in faceRects:
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
    # for (fX, fY, fW, fH) in pfaceRects:
    #     cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 255), 2)
    cv2.imshow("Face", frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
