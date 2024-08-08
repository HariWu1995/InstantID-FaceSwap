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

from api.model_util import load_models_xl, get_torch_device, torch_gc
from api.style_template import styles
from pipeline.sdxl_instantid_full import StableDiffusionXLInstantIDPipeline as SdXlInstantIdPipeline


# global variable
MAX_SEED = np.iinfo(np.int32).max

device = get_torch_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

STYLE_NAMES = list(styles.keys())
STYLE_DEFAULT = "(No style)"


# functions
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


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


def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[STYLE_DEFAULT])
    return p.replace("{prompt}", positive), n + ' ' + negative


def generate_image( face_image, 
                    pose_image, 
                        prompt, 
               negative_prompt, 
                    style_name, 
                     num_steps, 
    identitynet_strength_ratio, 
        adapter_strength_ratio, 
                guidance_scale, 
                          seed, 
                    enable_LCM, 
           enhance_face_region, MODEL_CONFIG):

    # Re-load Model every query, to save memory
    face_encoder_dir = MODEL_CONFIG.get('face_encoder_dir', './')
    face_adapter_path = MODEL_CONFIG.get('face_adapter_path', './checkpoints/ip-adapter.bin')
    controlnet_path = MODEL_CONFIG.get('controlnet_path', './checkpoints/ControlNetModel')
    sdxl_ckpt_path = MODEL_CONFIG.get('sdxl_ckpt_path', 'wangqixun/YamerMIX_v8')
    lora_ckpt_path = MODEL_CONFIG.get('lora_ckpt_path', 'latent-consistency/lcm-lora-sdxl')
    
    # Preprocess face
    # face_image = load_image(face_image_path)
    face_image = resize_img(face_image)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    # Load face encoder
    print("\nLoading Face Encoder ...")
    app = FaceAnalysis(name='antelopev2', root=face_encoder_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Extract face features
    print("\nExtracting targeted face features ...")
    face_info = app.get(face_image_cv2)
    
    if len(face_info) == 0:
        raise gr.Error(f"Cannot find any face in the image! Please upload another person image")
    
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info['kps'])
    
    if pose_image is not None:
        # pose_image = load_image(pose_image_path)
        pose_image = resize_img(pose_image)
        pose_image_cv2 = convert_from_image_to_cv2(pose_image)
        
        print("\nExtracting referenced face features ...")
        face_info = app.get(pose_image_cv2)
        
        if len(face_info) == 0:
            raise gr.Error(f"Cannot find any face in the reference image! Please upload another person image")
        
        face_info = face_info[-1]
        face_kps = draw_kps(pose_image, face_info['kps'])
        
        width, height = face_kps.size

    # Remove face detector to save memory
    del app

    if enhance_face_region:
        print("\nEnhancing face ...")
        control_mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = face_info["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None
                    
    generator = torch.Generator(device=device).manual_seed(seed)

    # Load pipeline
    print("\nLoading ControlNet ...")
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)

    print("\nLoading SD-XL Instant-Id Pipeline ...")
    if sdxl_ckpt_path.endswith(".ckpt") \
    or sdxl_ckpt_path.endswith(".safetensors"):

        scheduler_kwargs = hf_hub_download(
            repo_id="wangqixun/YamerMIX_v8",
            subfolder="scheduler",
            filename="scheduler_config.json",
        )
        scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)

        (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
            pretrained_model_name_or_path=sdxl_ckpt_path,
            scheduler_name=None,
            weight_dtype=dtype,
        )

        pipe = SdXlInstantIdPipeline(
            vae=vae,
            unet=unet,
            text_encoder=text_encoders[0],
            text_encoder_2=text_encoders[1],
            tokenizer=tokenizers[0],
            tokenizer_2=tokenizers[1],
            scheduler=scheduler,
            controlnet=controlnet,
        ).to(device)

    else:
        pipe = SdXlInstantIdPipeline.from_pretrained(
            sdxl_ckpt_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    print("\nLoading Instant-Id IP-Adapter ...")
    pipe.load_ip_adapter_instantid(face_adapter_path)

    # save VRAM
    print("\nEnabling CPU Offload ...")
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    # load and disable LCM LoRA
    if enable_LCM:
        print("\nEnabling LoRA ...")
        if lora_ckpt_path.endswith('safetensors'):
            lora_dir, ckpt_name = os.path.split(lora_ckpt_path)
            pipe.load_lora_weights(lora_dir, weight_name=ckpt_name)
        else:
            pipe.load_lora_weights(lora_ckpt_path)
        pipe.enable_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        print("\nDisabling LoRA ...")
        pipe.disable_lora()
        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    
    if prompt is None:
        prompt = "a person"
    
    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    print("\nInferencing ...")
    print(f"\t[Debug] Prompt: {prompt}, \n\t[Debug] Neg Prompt: {negative_prompt}")
    
    images = pipe(
                           prompt = prompt,
                  negative_prompt = negative_prompt,
                     image_embeds = face_emb,
                            image = face_kps,
                     control_mask = control_mask,
    controlnet_conditioning_scale = float(identitynet_strength_ratio),
              num_inference_steps = num_steps,
                   guidance_scale = guidance_scale,
                           height = height,
                            width = width,
                        generator = generator
    ).images

    # Clean
    del pipe, controlnet
    if 'cuda' in device:
        torch.cuda.empty_cache()

    return images[0]



    