# -*- coding:utf-8 -*-
import cv2


# 输入清晰图像的灰度图，返回对应模糊图像的灰度图
def get_blur_image(img):
    height, width = img.shape
    # print('Distorting image...')
    distorted = img.copy()
    distorted = cv2.resize(distorted, (width * 3, height * 3), interpolation=cv2.INTER_LINEAR)
    distorted = cv2.resize(distorted, (width, height), interpolation=cv2.INTER_LINEAR)
    noise_num = distorted.size / 10
    # for i in range(noise_num):  # 添加点噪声
    #     temp_x = np.random.randint(0, distorted.shape[0])
    #     temp_y = np.random.randint(0, distorted.shape[1])
    #     distorted[temp_x][temp_y] += 0.2 * np.random.randn(1, 1)
    # distorted[:, :] += 0.15 * np.random.randn(height, width)
    distorted = cv2.GaussianBlur(distorted, (3, 3), 0)
    distorted = cv2.GaussianBlur(distorted, (7, 7), 0)
    # cv2.imshow('distorted', distorted)
    # cv2.waitKey(0)
    return distorted