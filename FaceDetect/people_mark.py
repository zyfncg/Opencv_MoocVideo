from __future__ import print_function
import imutils
from facedetector import FaceDetector
import cv2

video_path = "picture/test2.flv"
choose = 0
if choose == 0:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(video_path)

(grabbed, frame) = camera.read()
fd = FaceDetector("haarcascade_upperbody.xml")
while True:
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    frameClone = frame.copy()

    for (fX, fY, fW, fH) in faceRects:
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 255, 0), 2)
    cv2.imshow("Face", frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
