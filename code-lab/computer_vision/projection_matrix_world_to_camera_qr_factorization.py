"""Pixel location of 2d project point `o_world` into image plane.

This is from a quiz from, `visual-perception-self-driving-cars` course in coursera.
Should be useful for those who have a hard time understanding this.

Projection matrix, P = K[R|t]O
World to camera vector, (Intrinsic matrix not needed) = [R|t]O
"""
import cv2.cv2 as cv2
import numpy as np

o_world = np.array([-.5, 1.5, 9, 1])
# Identity matrix rotated around X to 180.
R_Xc180 = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
t = [1, 2, 10]
Rt = np.c_[R_Xc180, t]
K = np.array([[640, 0, 640], [0, 480, 480], [0, 0, 1]])
P = np.matmul(K, Rt)

print(K.shape, Rt.shape)
print(P)
print(P.shape)
# Pixel location on image plane.
print(P.dot(o_world))
# World to camera coordinate.
print(Rt.dot(o_world))

# QR decomposition/projection of projection matrix.
# One possible solution of non unique solution.
K_new, R_new, t_new, _, _, _, _ = cv2.decomposeProjectionMatrix(P.astype(np.float32))
t_new = t_new / t_new[3]
print('Calculated:')
print(K_new)
print(R_new)
print(t_new)
