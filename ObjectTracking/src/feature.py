import numpy as np
import matplotlib.pyplot as plt
import cv2

path = "../file/test3.png"
image = cv2.imread(path)
cv2.imshow("source", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradient = cv2.add(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

blurred = cv2.blur(gradient, (7, 7))
# (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
# hist_cv = cv2.calcHist([blurred], [0], None, [256], [0, 256])
# plt.plot(hist_cv)
# plt.show()

closed = cv2.erode(blurred, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

# (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(_, thresh1) = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("result1", thresh)
cv2.imshow("result2", thresh1)
cv2.imshow("result", blurred)
cv2.waitKey(0)
