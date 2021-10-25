"""Hungarian algortihm for cost maximization.

This is used for object association for tracking. The values here are IOU that is in [0,1].
Row direction here is for current detections and column direction is previously tracked objects.

The values come from a numpy vectorized function that takes two ndarrays where first is current
detection and second one is previous track. The array formats are in [N, 4]. Goal is to get the
indexes from detection that maximizes the Intersection Over Union(IOU) value.
"""

from scipy.optimize import linear_sum_assignment
import numpy as np


a = [
    [0., 0.71437076, 0.],
    [0., 0., 0.84353033],
]

b = [
    [0., 0.,],
    [0.68328221, 0.,],
    [0., 0.83312102],
]

c = [
    [0.91681661, 0., 0.,],
    [0., 0.77371833, 0.,],
    [0., 0., 0.84160564],
]

d = [
    [0.6, 0.71437076, 0.],
    [0.3, 0.2, 0.84353033],
]

e = [
    [0.9, 0.1,],
    [0.68328221, 0.95,],
    [0.7, 0.83312102],
]

f = [
    [0.91681661, 0.8, 0.,],
    [0., 0.77371833, 0.9,],
    [0.98, 0.7, 0.84160564],
]


def func(cost_arr):
    # If less than IOU threshold, not needed.
    # cost_arr[cost_arr < 0.4] = 0

    row_ind, col_ind = linear_sum_assignment(np.array(-cost_arr))
    return row_ind, col_ind


def get_details(arr):
    print("COST ARRAY:")
    arr = np.array(arr)
    print(arr)
    print("[DETECTION INDEX] [TRACKED INDEX]:")
    row_ind, col_ind = func(arr)
    print(row_ind, col_ind)
    print("MAX COST:")
    print(arr[row_ind, col_ind].sum())


if __name__ == '__main__':
    get_details(e)