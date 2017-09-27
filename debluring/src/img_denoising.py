import text_localization
import cv2
import numpy as np
from time import time
from sklearn.decomposition import SparseCoder
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

dicts = np.load('dict.npz')
blur_dict = dicts['arr_0']
clear_dict = dicts['arr_1']
original_dict = dicts['arr_2']
patch_size = (7, 7)


def block_denoise(img_box, method=0):
    img_r = img_box[:, :, 0]
    img_g = img_box[:, :, 1]
    img_b = img_box[:, :, 2]

    h, w = img_r.shape

    global blur_dict, clear_dict

    r_blur_patch = extract_patches_2d(img_r, patch_size)
    g_blur_patch = extract_patches_2d(img_g, patch_size)
    b_blur_patch = extract_patches_2d(img_b, patch_size)

    r_blur_patch = r_blur_patch.reshape(r_blur_patch.shape[0], -1)
    g_blur_patch = g_blur_patch.reshape(g_blur_patch.shape[0], -1)
    b_blur_patch = b_blur_patch.reshape(b_blur_patch.shape[0], -1)

    r_mean = np.mean(r_blur_patch, axis=0)
    g_mean = np.mean(g_blur_patch, axis=0)
    b_mean = np.mean(b_blur_patch, axis=0)
    r_blur_patch -= r_mean
    g_blur_patch -= g_mean
    b_blur_patch -= b_mean
    r_blur_patch /= np.std(r_blur_patch, axis=0)
    g_blur_patch /= np.std(g_blur_patch, axis=0)
    b_blur_patch /= np.std(b_blur_patch, axis=0)

    if method == 0:
        coder_dict = blur_dict
        trans_dict = clear_dict
    else:
        coder_dict = original_dict
        trans_dict = original_dict
    coder = SparseCoder(dictionary=coder_dict, transform_n_nonzero_coefs=3,
                        transform_alpha=None, transform_algorithm='omp')
    r_code = coder.transform(r_blur_patch)
    g_code = coder.transform(g_blur_patch)
    b_code = coder.transform(b_blur_patch)
    r_clear_patch = np.dot(r_code, trans_dict)
    g_clear_patch = np.dot(g_code, trans_dict)
    b_clear_patch = np.dot(b_code, trans_dict)
    r_clear_patch += r_mean
    g_clear_patch += g_mean
    b_clear_patch += b_mean
    r_clear_patch = r_clear_patch.reshape(len(r_blur_patch), *patch_size)
    g_clear_patch = g_clear_patch.reshape(len(g_blur_patch), *patch_size)
    b_clear_patch = b_clear_patch.reshape(len(b_blur_patch), *patch_size)

    r_clear_block = reconstruct_from_patches_2d(r_clear_patch, (h, w))
    g_clear_block = reconstruct_from_patches_2d(g_clear_patch, (h, w))
    b_clear_block = reconstruct_from_patches_2d(b_clear_patch, (h, w))

    new_image_box = img_box.copy()
    new_image_box[:, :, 0] = r_clear_block
    new_image_box[:, :, 1] = g_clear_block
    new_image_box[:, :, 2] = b_clear_block
    contrast_r = np.hstack((img_r, r_clear_block))
    contrast_g = np.hstack((img_g, g_clear_block))
    contrast_b = np.hstack((img_b, b_clear_block))
    contrast = np.vstack((contrast_r, contrast_g))
    contrast = np.vstack((contrast, contrast_b))
    # cv2.imshow('contrast', contrast)
    # cv2.waitKey(0)
    return new_image_box


def image_denoise(image_path):
    print('start...')
    t0 = time()
    image = cv2.imread(image_path)
    text_boxes = text_localization.text_detect(image)
    new_image = image.copy() / 255.0
    for index_, (contour_, box) in enumerate(text_boxes):
        x, y, w, h = box
        text_block = new_image[y:y + h, x:x + w, :]
        clear_block = block_denoise(text_block, 1)
        new_image[y:y + h, x:x + w] = clear_block
        # cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print('done in %.2fs.' % (time() - t0))
    cv2.imshow('original', image)
    cv2.imshow('result', new_image)
    cv2.waitKey(0)


path = '../test_data/111.png'
image_denoise(path)
