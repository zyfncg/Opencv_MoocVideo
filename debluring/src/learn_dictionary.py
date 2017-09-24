# -*- coding:utf-8 -*-
from time import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder
from sklearn.decomposition import dict_learning
from sklearn.feature_extraction.image import extract_patches_2d
import text_localization

blur_patches = []
original_patches = []
patch_size = (8, 8)


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


# 获得用于训练的模糊和清晰图像对应得patch块
def get_image_patches(image_path):
    global original_patches, blur_patches
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    img_gray = img_gray / 255.0
    text_boxes = text_localization.text_detect(img)
    # print (text_boxes)
    temp_original_patches = []
    temp_blur_patches = []
    for index_, (contour_, box) in enumerate(text_boxes):
        x, y, w, h = box
        text_block = img_gray[y:y+h, x:x+w]
        blur_block = get_blur_image(text_block)
        clear_patch = extract_patches_2d(text_block, patch_size)
        blur_patch = extract_patches_2d(blur_block, patch_size)

        # 获得文本块的patches
        clear_patch = clear_patch.reshape(clear_patch.shape[0], -1)
        blur_patch = blur_patch.reshape(blur_patch.shape[0], -1)

        if len(temp_original_patches) == 0 and len(temp_blur_patches) == 0:
            temp_original_patches = clear_patch
            temp_blur_patches = blur_patch
        else:
            temp_original_patches = np.vstack((temp_original_patches, clear_patch))
            temp_blur_patches = np.vstack((temp_blur_patches, blur_patch))

        # cv2.imshow("block", text_block)
        # cv2.waitKey(0)

    # 添加到总样本中
    if len(original_patches) == 0 and len(blur_patches) == 0:
        original_patches = temp_original_patches
        blur_patches = temp_blur_patches
    else:
        original_patches = np.vstack((original_patches, temp_original_patches))
        blur_patches = np.vstack((blur_patches, temp_blur_patches))


# 计算所有的训练的模糊和清晰图像对应得patch块
def get_all_train_patches():

    dir_path = '../train_data/'
    img_names = ['timg.jpg', 'text1.jpg']
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 灰度化
    for name in img_names:
        get_image_patches(dir_path+name)
    print('get %d patches' % (blur_patches.shape[0]))


def train_patches():
    global blur_patches, original_patches

    get_all_train_patches()
    blur_patches -= np.mean(blur_patches, axis=0)
    blur_patches /= np.std(blur_patches, axis=0)
    original_patches -= np.mean(original_patches, axis=0)
    original_patches /= np.std(original_patches, axis=0)
    print('Learning the dictionary...')
    t0 = time()
    dico = MiniBatchDictionaryLearning(n_components=256, alpha=1, n_iter=200)
    V = dico.fit(blur_patches).components_
    # (code, V, err, n_iter) = dict_learning(X=blur_patches, n_components=256, alpha=1, max_iter=200)
    np.savez('dict.npz', V)
    dt = time() - t0
    print('done in %.2fs.' % dt)

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[0:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary learned from text patches\n' +
                 'Train time %.1fs on %d patches' % (dt, len(blur_patches)),
                 fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    print('compute clear Dict...')
    t0 = time()
    coder = SparseCoder(dictionary=V, transform_n_nonzero_coefs=1, transform_alpha=None, transform_algorithm='omp')
    code = coder.transform(blur_patches)
    code = np.array(code)
    print('code size is %d  %d' % (code.shape[0], code.shape[1]))
    temp = np.linalg.pinv(np.dot(code.T, code))
    clear_dict = np.dot(np.dot(original_patches.T, code), temp)
    clear_dict = clear_dict.T
    np.savez('dict.npz', V, clear_dict)
    dt = time() - t0
    print('done in %.2fs.' % dt)

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(clear_dict[0:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary learned from text patches\n' +
                 'Train time %.1fs on %d patches' % (dt, len(blur_patches)),
                 fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()


train_patches()
