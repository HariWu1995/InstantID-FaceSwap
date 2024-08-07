import sys
sys.path.append('./')

from typing import Tuple

import os
import cv2
import math
import torch
import random
import numpy as np
import argparse

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler

from huggingface_hub import hf_hub_download

import insightface
from insightface.app import FaceAnalysis

from model_util import load_models_xl, get_torch_device, torch_gc
from style_template import styles
from pipeline.sdxl_instantid_full import StableDiffusionXLInstantIDPipeline as SdXlInstantIdPipeline


#######################
#   Global Variables  #
#######################

MAX_SEED = np.iinfo(np.int32).max

device = get_torch_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

STYLE_NAMES = list(styles.keys())
STYLE_DEFAULT = "Watercolor"


#######################
#      Utilities      #
#######################

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
            pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):

        w, h = input_image.size
        if size is not None:
            w_resize_new, h_resize_new = size
        else:
            ratio = min_side / min(h, w)
            w, h = round(ratio*w), round(ratio*h)
            ratio = max_side / max(h, w)
            input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
            w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
            h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
        input_image = input_image.resize([w_resize_new, h_resize_new], mode)

        if pad_to_max_side:
            res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
            offset_x = (max_side - w_resize_new) // 2
            offset_y = (max_side - h_resize_new) // 2
            res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
            input_image = Image.fromarray(res)
        return input_image

def apply_style(positive: str, negative: str = "", style_name: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[STYLE_DEFAULT])
    return p.replace("{prompt}", positive), n + ' ' + negative


#######################
#         API         #
#######################

from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()




