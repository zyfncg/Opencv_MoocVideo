#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017.11.11 19:49
# @Author  : Aries
# @Site    : 
# @File    : png2bmp.py
# @Software: PyCharm
import os
import cv2


def gen_bmp_dataset(image_path_dir,target_path_dir):
    img_names = os.listdir(image_path_dir)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 灰度化
    for name in img_names:
        print "handle:"+image_path_dir+name+">>>"+target_path_dir+name.split(".")[0]+".bmp"
        if name.split(".")[1]=="png":
            jpgImg=cv2.imread(image_path_dir+name)
            name=name.split(".")[0]+".bmp"
            cv2.imwrite(image_path_dir+name,jpgImg)
        # sourceImage = cv2.imread(image_path_dir+name)
        # raw_name=target_path_dir+name.split(".")[0]
        # cv2.imwrite(raw_name+"_0."+name.split(".")[1],blur_guass)
        # cv2.imwrite(raw_name + "_1."+name.split(".")[1], blur_plain)

src_path="../all_img/"
dest_path="../all_img_bmp/"
gen_bmp_dataset(src_path,dest_path)