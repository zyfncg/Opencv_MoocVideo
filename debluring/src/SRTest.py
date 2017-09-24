# -*- coding:utf-8 -*-
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
import cv2
import scipy as sp

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


text = cv2.imread('../train_data/timg.jpg')
text = text / 255.0
img_r = text[:, :, 0]
img_g = text[:, :, 1]
img_b = text[:, :, 2]
# print img_r[:5, :6]
# print img_g[:5, :6]
# print img_b[:5, :6]
height, width = img_r.shape
print('Distorting image...')
distorted = img_r.copy()
distorted = cv2.resize(distorted, (width*2, height*2), interpolation=cv2.INTER_LINEAR)
distorted = cv2.resize(distorted, (width, height), interpolation=cv2.INTER_LINEAR)
noise_num = distorted.size/10
for i in range(noise_num):  # 添加点噪声
    temp_x = np.random.randint(0, distorted.shape[0])
    temp_y = np.random.randint(0, distorted.shape[1])
    distorted[temp_x][temp_y] += 0.2*np.random.randn(1, 1)
distorted[:, :] += 0.15 * np.random.randn(height, width)
distorted = cv2.GaussianBlur(distorted, (3, 3), 0)
distorted = cv2.GaussianBlur(distorted, (5, 5), 0)
print('Extracting reference patches...')
t0 = time()
patch_size = (8, 8)
dataL = extract_patches_2d(distorted[:, :], patch_size)
dataH = extract_patches_2d(img_r.copy()[:, :], patch_size)
print(dataL.shape[0])
dataL = dataL.reshape(dataL.shape[0], -1)
dataH = dataH.reshape(dataH.shape[0], -1)
# train_data -= np.mean(train_data, axis=0)
# train_data /= np.std(train_data, axis=0)
print(dataL.shape)
print('done in %.2fs.' % (time() - t0))

# #############################################################################
# Learn the dictionary from reference patches

print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=256, alpha=1, n_iter=200)
V = dico.fit(dataL).components_
dt = time() - t0
print('done in %.2fs.' % dt)

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap='gray',
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from text patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(dataL)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

print('compute HS Dict...')
t0 = time()
coder = SparseCoder(dictionary=V, transform_n_nonzero_coefs=3, transform_alpha=None, transform_algorithm='omp')
# code = coder.transform(dataL)
# code = np.array(code)
# temp = np.dot(code, code.T)
# temp = temp.I
# Dh = np.dot(np.dot(dataH, code.T), temp)
dt = time() - t0
print('done in %.2fs.' % dt)


# train_data = extract_patches_2d(distorted[:, width // 2:], patch_size)
# train_data = train_data.reshape(train_data.shape[0], -1)
# code = dico.transform(train_data)
# patches = np.dot(code, Dh)
# patches = patches.reshape(len(train_data), *patch_size)
# reconstructions = distorted.copy()
# reconstructions[:, width // 2:] = reconstruct_from_patches_2d(
#     patches, (height, width // 2))
plt.figure(1)
plt.imshow(text)
plt.figure(2)
plt.imshow(distorted, cmap='gray')
# plt.figure(3)
# plt.imshow(reconstructions, cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.show()
