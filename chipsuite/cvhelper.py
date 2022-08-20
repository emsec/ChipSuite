#!/usr/bin/env python3
import cv2
import numpy as np


def cvStampImageScaled(dest, src, dx, dy, w, h):
    scaled = cv2.resize(src, (w, h))
    sx = 0
    dx1 = dx
    dx2 = dx + w
    sy = 0
    dy1 = dy
    dy2 = dy + h
    if dx1 < 0:
        sx -= dx1
        dx1 = 0
    if dy1 < 0:
        sy -= dy1
        dy1 = 0
    if dy2 > dy1 and dx2 > dx1:
        dest[dy1:dy2,dx1:dx2] = scaled[sy:,sx:]
    return dest


def imshowx(name, image, x, y, resizeable=False):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL if resizeable else cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(name, x, y)
    cv2.imshow(name, image)


def imshowy(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ball_template(size=11, width=None, height=None):
    #assert size % 2, "Ball size is required to be odd"
    radius = (size + 1) // 2
    ballheight = height if height else size
    ballwidth = width if width else size
    ball = np.zeros((ballheight, ballwidth), np.uint8)
    cx = ballwidth // 2
    cy = ballheight // 2
    for i in range(radius):
        color = round((i + 1) / radius * 255)
        cv2.circle(ball, (cx, cy), radius-i, color, -1)
    return ball