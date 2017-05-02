import cv2  
import numpy as np  
  
img = cv2.imread("111.png")
emptyImage = np.zeros(img.shape, np.uint8)  
emptyImage[0:10,0:100]=(255,0,0)
cv2.imshow("img1",emptyImage)
emptyImage2 = img.copy()  
 
emptyImage3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
  
cv2.imshow("EmptyImage3", emptyImage3)  
cv2.waitKey (0)  
cv2.destroyAllWindows()