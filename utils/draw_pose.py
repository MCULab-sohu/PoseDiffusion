from matplotlib import pyplot as plt
import torch
import os
import numpy as np
import cv2
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get_coors_via_mask
def get_coor(image):
    data = image.cpu().detach().numpy()
    area_threshold = 5
    coordinates = []
    for idx in range(data.shape[0]):
        img = data[idx]
        sorted_pixels = np.sort(img.flatten())
        threshold_value = sorted_pixels[int(0.05 * len(sorted_pixels))]

        _, mask = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_large_area = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                has_large_area = True
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                break
        if not has_large_area:
            cx = 0
            cy = 0
        coordinates.append((cx,cy))
    return coordinates
# get_subs_new
def get_subs(channel_num, coordinates):
    channel_num = channel_num
    coordinates = coordinates
    subset = []
    for channel in range(channel_num):
        if coordinates[channel]==(0,0):
            subset.append(-1)
        else:
            subset.append(channel)
    return subset
# draw body
def draw_bodypose(candidate, subset):
    stickwidth = 5
    candidate = candidate * 512/32
    canvas_shape = np.zeros((512, 512, 3))
    limbSeq = [[1, 2], [1, 3], [2, 4], [3, 5], [6, 7], [6, 8], [6, 12], [7, 9], \
               [7, 13], [8, 10], [9, 11], [12, 13], [12, 14], [13, 15], [14, 16], \
               [15, 17], [4, 6], [5, 7]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0]]
    canvas = np.zeros_like(canvas_shape)
    for i in range(17):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(16):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas