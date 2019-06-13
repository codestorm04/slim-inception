import os
from PIL import Image
import numpy as np
import cv2

# Crop the only object in a figure
def bounding_crop(image, rows, cols):
    img = cv2.Canny(image, 120, 200)

    tmp = np.sum(img, axis=0)  
    x, w = _hill_filter(tmp)  
    tmp = np.sum(img, axis=1)
    y, h = _hill_filter(tmp)  

    if x is not None and y is not None and w is not None and h is not None:
        x1, y1, w1, h1 = _aspect_ratio_rectify(x, y, w, h, img.shape[1], img.shape[0])
        image = image[y1: y1+h1, x1: x1+w1, : ]
        image = cv2.resize(image, (rows, cols))

        # map back to origal coordinator
        # ratio = img.shape[0] / rows
        # y = int(y * ratio)
        # h = int(h * ratio)
        # ratio = image.shape[1] / cols
        # x = int(x * ratio)
        # w = int(w * ratio)
    else:
        x = 0
        y = 0
        w = image.shape[0]
        h = image.shape[1]

    # cv2.imwrite(filename, image)
    # cv2.imshow(str(random.random()), img)
    # cv2.imshow(str(random.random()), image) 
    return image, x, y, w, h

# filer the small hills of accumulative array to remove noise points
def _hill_filter(arr):
    arr = arr/255
    acc = np.empty([0, 3], dtype=int)
    tmp = 0
    start = -1
    end = -1
    for i in range(len(arr)):
        if arr[i] > 0:
            tmp += arr[i]
            if start == -1:
                start = i
            if i == len(arr) - 1:
                end = i
                acc = np.concatenate((acc, [np.array([tmp, start, end])]), axis=0)
        elif tmp != 0:
            end = i - 1
            acc = np.concatenate((acc, [np.array([tmp, start, end])]), axis=0)
            tmp = 0
            start = -1
    if len(acc) > 0:
        thresh = np.mean(acc, axis = 0)[0] * 0.15
        x = 0
        x2 = 0
        for hill in acc:
            if hill[0] > thresh:
                x = hill[1]
                break
        for hill in acc[::-1, :]:
            if hill[0] > thresh:
                x2 = hill[2]
                return int(x), int(x2 - x)
    return None, None

# If the aspect ratio more than ratio_thresh then keep padding
def _aspect_ratio_rectify(x, y, w, h, width, height):
    ratio_thresh = 0.667
    if w > h * ratio_thresh and h > w * ratio_thresh:
        return x, y, w, h
    if w <= h * ratio_thresh:
        delta = int((h * ratio_thresh - w) / 2)
        w = int(h * ratio_thresh)
        x = max(0, x - delta)
        w = min(w, width)
    else:
        delta = int((w * ratio_thresh - h) / 2)
        h = int(w * ratio_thresh)
        y = max(0, y - delta)
        h = min(h, width)
    return x, y, w, h


    
if __name__ == '__main__':
    # load_js_data()

    # resizeImg(path='goodsdata/data_all/', img_rows=300, img_cols=300)
    # load_js_data(path='JS_Data/', img_rows=200, img_cols=320)
    bounding_crop(path="/home/lyz/desktop/goods_data_croped/", img_rows=299, img_cols=299)