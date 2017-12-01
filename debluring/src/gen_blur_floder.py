#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017.11.8 19:56
# @Author  : Aries
# @Site    : 
# @File    : gen_blur_floder.py
# @Software: PyCharm
import cv2
import numpy as np
import os

def get_blur_image_guass(img):
    # learn_dictionary是二维图像，img_bluring是三维，记得切换
    height, width, modelen = img.shape
    distorted = img.copy()
    distorted = cv2.resize(distorted, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
    distorted = cv2.resize(distorted, (width, height), interpolation=cv2.INTER_CUBIC)
    noise_num = distorted.size / 10
    # 加噪声并没有什么用
    # 3-7-34.5 5-7-34.2 7-9-33.5 9-11-33.1 11-13-32.7 11-13-15-32.0
    distorted = cv2.GaussianBlur(distorted, (5, 5), 0)
    distorted = cv2.GaussianBlur(distorted, (7, 7), 0)
    # distorted = cv2.GaussianBlur(distorted, (15, 15), 0)
    return distorted

# 生成的5*5核模板其实就是一个均值滤波
def get_blur_image_plain(img):
    height, width, mode_len = img.shape
    distorted = img.copy()
    distorted = cv2.resize(distorted, (width * 3, height * 3), interpolation=cv2.INTER_LINEAR)
    distorted = cv2.resize(distorted, (width, height), interpolation=cv2.INTER_LINEAR)
    # 15-31.7 10-32.7 5-34.6
    # 有重影
    distorted = cv2.blur(img, (5, 5))
    return distorted

def gen_dataset(image_path_dir,target_path_dir):
    img_names = os.listdir(image_path_dir)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 灰度化
    for name in img_names:
        print "handle:"+image_path_dir+name+">>>"+target_path_dir+name
        if name.split(".")[1]=="jpg":
            jpgImg=cv2.imread(image_path_dir+name)
            name=name.split(".")[0]+".png"
            cv2.imwrite(image_path_dir+name,jpgImg)
        sourceImage = cv2.imread(image_path_dir+name)
        blur_guass = get_blur_image_guass(sourceImage)
        blur_plain = get_blur_image_plain(sourceImage)
        raw_name=target_path_dir+name.split(".")[0]
        cv2.imwrite(raw_name+"_0."+name.split(".")[1],blur_guass)
        cv2.imwrite(raw_name + "_1."+name.split(".")[1], blur_plain)
src_dir="../all_img/"
dest_dir="../output_img/"
gen_dataset(src_dir,dest_dir)