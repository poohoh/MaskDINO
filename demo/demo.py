# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import random

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    # parser.add_argument(
    #     "--input",
    #     nargs="+",
    #     help="A list of space separated input images; "
    #     "or a single glob pattern such as 'directory/*.jpg'",
    # )
    parser.add_argument(
        "--output",
        default="output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'weights/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_mask52.1ap_box58.3ap.pth'],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--root_dir', default='C:/Users/KimJunha/Desktop/test/car', type=str, help='image root dir')
    parser.add_argument('--bg_dir', default='C:/Users/KimJunha/Desktop/test/background', type=str)

    return parser

def composite_with_mask(image, mask, bg_path):
    # choose random background
    backgrounds = glob.glob(os.path.join(bg_path, '*.png'))
    background = random.choice(backgrounds)
    background = cv2.imread(background)

    # object only
    object_only = cv2.bitwise_and(image, image, mask=mask)

    # background only
    inverse_mask = cv2.bitwise_not(mask)
    background_only = cv2.bitwise_and(background, background, mask=inverse_mask)

    # composite background
    composite_result = cv2.add(object_only, background_only)

    return object_only, background_only, composite_result

# 차량의 번호 (e.g., 000001, 000002 (4), ...)
def get_car_num(path):
    result = path.split(os.sep) if os.sep in path else path.split('/')
    result = result[-2].split('_')[-1]
    return result

def get_biggest_mask(masks):
    # remove masks bigger than 4000
    areas = np.array([np.sum(mask) for mask in masks])
    inds = np.where(areas < 3800)[0]
    areas = areas[inds]
    masks = masks[inds]

    # biggest mask
    mask_main = masks[np.argmax(areas)]
    mask_main = (mask_main * 255).astype(np.uint8)

    return mask_main

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # spawn: 부모 프로세스가 새로운 프로세스 시작 후 자식 프로세스 호출, force: 현재 설정된 시작 방법을 무시하고 새로운 방법 강제 설정
    args = get_parser().parse_args()

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    dirs = [args.root_dir]

    for dir in dirs:
        dir_inside = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
        dirs.extend(dir_inside)

        # paths
        img_paths = glob.glob(os.path.join(dir, '*.png'))

        print('processing:', dir)

        for img_path in tqdm.tqdm(img_paths, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(img_path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)

            # get biggest mask
            masks = predictions['instances'].get('pred_masks').cpu().numpy()
            if len(masks) < 1:
                continue
            mask_main = get_biggest_mask(masks)

            # composite background
            object_only, background_only, composite_result = composite_with_mask(img, mask_main, args.bg_dir)

            # save results
            car_num = get_car_num(img_path)
            img_idx = os.path.splitext(os.path.basename(img_path))[0]
            mask_dir = os.path.join(args.root_dir, '..', 'mask_result', car_num)
            composite_dir = os.path.join(args.root_dir, '..', 'composite_result', car_num)
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(composite_dir, exist_ok=True)

            cv2.imwrite(os.path.join(mask_dir, f'{img_idx}_mask.png'), mask_main)
            cv2.imwrite(os.path.join(mask_dir, f'{img_idx}_object.png'), object_only)
            cv2.imwrite(os.path.join(composite_dir, f'{img_idx}_composite.png'), composite_result)