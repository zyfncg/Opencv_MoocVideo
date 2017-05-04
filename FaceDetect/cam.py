from __future__ import print_function
from facedetector import FaceDetector
import imutils
import cv2


video_path = "picture/test1.flv"
choose = 1
if choose == 0:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(video_path)

fd = FaceDetector("haarcascade_frontalface_default.xml")
while True:
    (grabbed, frame) = camera.read()
    #print(grabbed)
    if choose == 1 and not grabbed:
        break
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #print("I found {} face(s)".format(len(faceRects)))

    frameClone = frame.copy()
    for (x, y, w, h) in faceRects:
        cv2.rectangle(frameClone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Faces", frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
