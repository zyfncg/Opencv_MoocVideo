#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/27/2017 21:33
# @Author  : OptimusPrime
# @Site    : 
# @File    : image_handle.py
# @Software: PyCharm Community Edition

import cv2
import math
import numpy as np
from PIL import ImageFilter
from PIL import ImageEnhance

def multiFilter(img):
  # img = Image.open(path)

##图像处理##
# 转换为RGB图像
  img = img.convert("RGB")
  ##组合使用filter,边界增强加强版+边界平滑+锐化+细节
  group_img1 = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
  group_img1 = group_img1.filter(ImageFilter.SHARPEN)
  group_img1 = group_img1.filter(ImageFilter.BLUR)
  group_img1 = group_img1.filter(ImageFilter.DETAIL)
  group_img1 = group_img1.filter(ImageFilter.SMOOTH_MORE)
  # pos=path.index('.')
  # tarPath=path[0:pos]+"_multi.png"
  # group_img1.save(tarPath)
  return group_img1


def multiOperation(srcImage):
    """
从亮度、色度、对比度、锐度等四个方面进行增强
    :param srcPath:
    """
    # srcImage=Image.open(srcPath)

    enh_bri=ImageEnhance.Brightness(srcImage)
    image_brightened = enh_bri.enhance(1.1)
    # image_brightened.save(srcPath)

    enh_col=ImageEnhance.Color(image_brightened)
    image_colored = enh_col.enhance(1.2)
    # image_colored.save(srcPath)

    enh_con=ImageEnhance.Contrast( image_colored)
    image_contrasted = enh_con.enhance(1.2)
    # image_contrasted.save(srcPath)

    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    # sharpness = 3.0
    image_sharpened = enh_sha.enhance(3.0)
    # pos=srcPath.index('.')
    # tarPath=srcPath[0:pos] + "_tar.png"
    # image_sharpened.save(tarPath)
    return image_sharpened

# def bothEnhance(img):
#     # filter_img=multiFilter(img)
#     # pos=srcPath.index('.')
#     # srcPath2=srcPath[0:pos]+'_multi.png'
#     res=multiOperation(img)
#     return res

def projectTransform(img, srcPoints):
    # index = source.index('.')
    # target = source[0:index] + "_project.png"
    # img = cv2.imread(source)
    # print "sourcePath:"+source
    rows, cols = img.shape[:2]
    print "Project===rows:", rows, "---cols:", cols

    pts1 = np.float32(srcPoints)
    pts2 = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], int(math.floor(img.shape[1] * 3 / 4))],
                       [0, int(math.floor(img.shape[1] * 3 / 4))]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # 第三个参数：变换后的图像大小，第一个为高度，第二个为宽度

    res = cv2.warpPerspective(img, M, (img.shape[1], int(math.floor(img.shape[1] * 3 / 4))))

    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(res)

    # cv2.imwrite(target, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return res

    # cv2.namedWindow(source)
    # cv2.imshow(source, img)
    #
    # cv2.namedWindow(target)
    # cv2.imshow(target, res)


# def handleImage(img2,array):
#     # source = "image1/image4.png"
#     # img2 = cv2.imread(source)
#     pts1 = np.float32(array)
#     pts2 = np.float32([[0, 0], [img2.shape[1], 0], [img2.shape[1], int(math.floor(img2.shape[1] * 3 / 4))],
#                        [0, int(math.floor(img2.shape[1] * 3 / 4))]])
#     projectTransform(img2, pts1, pts2)
    # pos=source.index('.')
    # proPath=source[0:pos]+"_project.png"
    # bothEnhance(proPath)

# if __name__=='__main__':
#     # multiFilter("image1/image4_project_tar.png")
#     # bothEnhance("image1/image4_project.png")
#     arrays=[[[156,6],[607,30],[612,485],[157,497]],
#     [[381,53],[808,64],[815,503],[380,512]],
#     [[219,5],[764,4],[768,506],[219,520]] ,
#     [[191, 46], [863, 37], [936, 560], [123, 563]],
#     [[142,14],[357,14],[362,171],[154,171]],
#     [[139,0],[354,0],[361,159],[146,160]],
#     [[85,112],[470,89],[455,439],[65,411]],
#     [[75,83],[390,85],[389,316],[77,317]],
#     [[164,112],[542,47],[560,410],[172,420]],
#     [[290,173],[786,162],[802,558],[277,549]] ];
#     # for i in range(0,10):
#     #     source="image1/image"+(i+1).__str__()+".png"
#     #     handleImage(source,arrays[i])




