from data_io import read_pfm, save_pfm
import os
import sys
import cv2
import numpy as np


def depth_evaluation(
    gt_depths_path,
    pred_depths_path,
):
    gt_depths = os.listdir(gt_depths_path)
    gt_depths.sort()
    pred_depths = os.listdir(pred_depths_path)
    pred_depths.sort()

    gt_depths_valid = []
    pred_depths_valid = []
    ratios = []
    num = len(gt_depths)
    for i in range(num):
        # read pfm
        gt_depth, _ = read_pfm(os.path.join(gt_depths_path, gt_depths[i]))
        pred_depth, _ = read_pfm(os.path.join(pred_depths_path, pred_depths[i]))

        # ratio = 1.43 / np.median(pred_depth)

        gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

        pred_depths_valid.append(pred_depth)
        gt_depths_valid.append(gt_depth)
        ratios.append(1)
    if not os.path.exists(os.path.join(pred_depths_path, "new_depth")):
        os.mkdir(os.path.join(pred_depths_path, "new_depth"))

    for i in range(len(pred_depths_valid)):
        gt_depth = gt_depths_valid[i]
        pred_depth = pred_depths_valid[i]

        pred_depth *= 0.1
        save_pfm(
            os.path.join(pred_depths_path, "new_depth", pred_depths[i]),
            pred_depth,
        )


if __name__ == "__main__":
    depth_ours_path = sys.argv[1]
    depth_nerf_path = sys.argv[2]
    depth_evaluation(depth_ours_path, depth_nerf_path)
