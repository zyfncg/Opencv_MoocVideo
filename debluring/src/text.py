import text_localization
import cv2


text_image = cv2.imread('../train_data/111.png')
text_boxes = text_localization.text_detect(text_image)
# print (text_boxes)
boxes = []
new_image = text_image.copy()
for index_, (contour_, box) in enumerate(text_boxes):
    boxes.append(box)

    x_, y_, width, height = box
    cv2.rectangle(new_image, (x_, y_), (x_ + width, y_ + height), (0, 255, 0), 2)
cv2.imshow('result', new_image)
cv2.waitKey(0)
print(boxes)
