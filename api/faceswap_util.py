import sys
sys.path.append('./')

from typing import Tuple

import os
import random
import argparse

import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont

import math
import numpy as np
import torch

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler

from huggingface_hub import hf_hub_download

import insightface
from insightface.app import FaceAnalysis

# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
import face_segmentation as FaceSeg

from api.model_util import load_models_xl, get_torch_device, torch_gc
from api.style_template import styles
from pipeline.sdxl_instantid_inpaint import StableDiffusionXLInstantIDInpaintPipeline as SdXlInstantIdPipeline


# global variable
MAX_SEED = np.iinfo(np.int32).max
DEVICE = get_torch_device()
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32

print(f"\n\nDevice = {DEVICE} - Dtype = {DTYPE}")

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
        res[offset_y:offset_y+h_resize_new, 
            offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def visualize_face_mask_and_keypoints(face_image: PIL.Image, 
                                      face_mask: np.ndarray = None, 
                                      face_keypts: np.ndarray = None):

    if face_mask is not None:
        face_mask = Image.fromarray(face_mask * 255, mode='L')
        overlay = Image.new('RGBA', face_image.size, (255, 255, 255, 0))
        drawing = ImageDraw.Draw(overlay)
        drawing.bitmap((0, 0), face_mask, fill=(255, 0, 0, 100))

        face_image = face_image.convert('RGBA')
        face_image.putalpha(200)
        face_image = Image.alpha_composite(face_image, overlay)

    if face_keypts is not None:
        font = ImageFont.truetype(r"C:\Windows\Fonts\comic.ttf", 19)
        drawing = ImageDraw.Draw(face_image)
        for pi, pt in enumerate(face_keypts.tolist()):
            # print(pi, pt)
            pt = np.array(pt)
            pt_ = tuple(list(pt-3) + list(pt+3))
            drawing.ellipse(xy=pt_, fill=(0, 0, 255), outline=(255, 255, 255), width=1)
            drawing.text(list(pt), str(pi+1), font=font, align ="right")

    return face_image


def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[STYLE_DEFAULT])
    p = p.format(prompt=positive)
    if len(negative) > 0:
        n = n + '. ' + negative
    return p, n


def get_bbox_from_mask(mask) -> Tuple[int, int, int, int]:

    if isinstance(mask, PIL.Image.Image):
        mask = convert_from_image_to_cv2(image)

    mask_boundaries = np.where(mask != 0)
    m_x1 = int(np.min(mask_boundaries[1]))
    m_x2 = int(np.max(mask_boundaries[1]))
    m_y1 = int(np.min(mask_boundaries[0]))
    m_y2 = int(np.max(mask_boundaries[0]))

    return m_x1, m_x2, m_y1, m_y2


def prepare_inputs( pose_image: PIL.Image.Image, 
                    mask_image: np.ndarray, 
                     face_info: dict,
                        resize: bool = True, 
                       padding: Tuple[int] = (20, 20), ):

    assert isinstance(pose_image, PIL.Image.Image), f"{pose_image.__class__} is not supported!"        
    assert isinstance(mask_image, np.ndarray), f"{mask_image.__class__} is not supported!"
    
    pad_W, pad_H = padding
    W, H = pose_image.size

    # Standardize
    mask_image = (mask_image * 255).astype(np.uint8)

    # Get mask bounding-box
    m_x1, m_x2, m_y1, m_y2 = get_bbox_from_mask(mask_image)

    # Get portrait bounding-box (mask + padding)
    p_x1 = max(0, m_x1 - pad_W)
    p_y1 = max(0, m_y1 - pad_H)
    p_x2 = min(W, m_x2 + pad_W)
    p_y2 = min(H, m_y2 + pad_H)
    p_x1, p_y1, p_x2, p_y2 = int(p_x1), int(p_y1), int(p_x2), int(p_y2)

    # Crop
    mask_image = Image.fromarray(mask_image)
    mask  = mask_image.crop((p_x1, p_y1, p_x2, p_y2))
    image = pose_image.crop((p_x1, p_y1, p_x2, p_y2))
    p_W, p_H = image.size

    # Resize image and keypoints
    kps = face_info['kps'] - [p_x1, p_y1]
    if resize:
        mask = resize_img(mask)
        image = resize_img(image)
        new_W, new_H = image.size
        kps *= [new_W / p_W, new_H / p_H]

    control_image = draw_kps(image, kps)

    return (mask, image, control_image), (p_x1, p_y1, p_W, p_H)


def swap_face_only( face_image, 
                    pose_image, 
                 mask_strength,
                mask_padding_W,
                mask_padding_H,
                mask_threshold,
                        prompt, 
               negative_prompt, 
                    style_name, 
                     num_steps, 
    identitynet_strength_ratio, 
     ip_adapter_strength_ratio, 
                guidance_scale, 
                          seed, 
                    enable_LCM, 
           enhance_face_region, MODEL_CONFIG, **kwargs):

    # Re-load Model every query, to save memory
    face_segmentor_dir = MODEL_CONFIG.get('face_segmentor_dir', './checkpoints')
    face_analyzer_dir = MODEL_CONFIG.get('face_analyzer_dir', './')
    face_adapter_path = MODEL_CONFIG.get('face_adapter_dir', './checkpoints') + '/ip-adapter.bin'
    controlnet_path = MODEL_CONFIG.get('face_adapter_dir', './checkpoints') + '/ControlNetModel'
    sdxl_ckpt_path = MODEL_CONFIG.get('sdxl_ckpt_path', 'wangqixun/YamerMIX_v8')
    lora_ckpt_path = MODEL_CONFIG.get('lora_ckpt_path', 'latent-consistency/lcm-lora-sdxl')
    
    # Re-load device
    cuda_device_id = MODEL_CONFIG.get('cuda_device_id', -1)
    if cuda_device_id != -1:
        device = torch.device(f'cuda:{cuda_device_id}')
        dtype = torch.float16
    else:
        device = DEVICE
        dtype = DTYPE
    
    # Preprocess face
    # face_image = load_image(face_image_path)
    # face_image = resize_img(face_image)
    face_image_arr = convert_from_image_to_cv2(face_image)
    
    # pose_image = load_image(pose_image_path)
    # pose_image = resize_img(pose_image)
    pose_image_arr = convert_from_image_to_cv2(pose_image)
    pose_width, pose_height = pose_image.size

    # Load face analyzer
    print(f"\nLoading Face Analyzer @ {face_analyzer_dir} ...")
    face_analyzer = FaceAnalysis(root=face_analyzer_dir, name='antelopev2', 
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    
    # Extract face features
    print("\nExtracting targeted face features ...")
    face_info = face_analyzer.get(face_image_arr)
    if len(face_info) == 0:
        raise ValueError(f"Cannot find any face in the targeted image! Please upload another person image")
    
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
    face_emb = face_info['embedding']
    
    print("\nExtracting referenced face features ...")
    face_info = face_analyzer.get(pose_image_arr)
    if len(face_info) == 0:
        raise ValueError(f"Cannot find any face in the reference image! Please upload another person image")
    
    face_area = lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
    face_info = sorted(face_info, key=face_area, reverse=True)[0]  # only use the largest face
    # face_kps = draw_kps(pose_image, face_info['kps'])

    face_bbox = face_info['bbox']
    left, top, right, bottom = face_bbox
    left, right = max(0, int(left-mask_padding_W)), min(pose_width, int(right+mask_padding_W))
    top, bottom = max(0, int(top-mask_padding_H)), min(pose_height, int(bottom+mask_padding_H))
    pose_image_face_size = (right-left, bottom-top)
    pose_image_face = pose_image.crop((left, top, right, bottom))
    pose_image_face = pose_image_face.resize((512, 820), PIL.Image.BILINEAR)
    pose_image_face.save('logs/bbox.png')

    # Load Face-Segmentation Model
    print(f"\nLoading Face-Segmentation @ {face_segmentor_dir} ...")
    face_segmentor = FaceSeg.model.FaceSegmentationNet()
    FaceSeg.utils.load_model_parameters(face_segmentor, params_dir=face_segmentor_dir)
    face_segmentor.eval()
    face_segmentor.to(device)
    
    # Automatic segmentation for face mask
    print("\nSegmenting inside bounding-box ...")
    with torch.no_grad():
        pose_image_face = face_segmentor.prepare(pose_image_face, normalize=True).unsqueeze(0)
        pose_image_face = pose_image_face.to(device)
        mask = face_segmentor(pose_image_face, as_pmap=True).detach().cpu().numpy()[0]
        mask = cv2.resize(mask, pose_image_face_size, interpolation = cv2.INTER_AREA if (bottom-top) > 820
                                                                else cv2.INTER_LINEAR)
    bbox_mask = (mask > mask_threshold).astype(np.uint8)
    face_mask = np.zeros(pose_image_arr.shape[:2], np.uint8)
    face_mask[top:bottom, left:right] = bbox_mask
    cv2.imwrite('logs/segment.png', face_mask * 255)

    # Noise removal
    print("\nRemoving noise in face mask ...")
    iters = 7
    kernel = np.ones((1, 1), np.uint8)
    face_mask = cv2.erode(face_mask, kernel, iterations=iters)
    face_mask = cv2.dilate(face_mask, kernel, iterations=iters)

    # Contour-and-Convex Filling
    print("\nFilling the largest contour ...")
    temp_mask = np.zeros(face_mask.shape, np.uint8)
    contours, _ = cv2.findContours(face_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    border = cv2.convexHull(contour, False)
    cv2.drawContours(temp_mask, [border], 0, 1, -1)
    cv2.imwrite('logs/contour.png', temp_mask * 255)

    # Smoothing & padding mask
    print("\nPadding mask ...")
    kernel = np.ones((mask_padding_H, mask_padding_W), np.uint8) 
    face_mask = cv2.medianBlur(temp_mask, 19)
    face_mask = cv2.dilate(face_mask, kernel, iterations=1)
    cv2.imwrite('logs/mask.png', face_mask * 255)

    # return visualize_face_mask_and_keypoints(pose_image, 
    #                                          face_mask, 
    #                                          face_info['landmark_2d_106'], )

    # Remove face analyzer & segmentor to save memory
    del face_analyzer, face_segmentor

    # Crop portrait from pose-image
    print("\nCropping portrait from pose-image ...")
    portrait_images, \
    portrait_coordinates = prepare_inputs(  pose_image, 
                                            face_mask,
                                            face_info, 
                                            resize=True,
                                            padding=(mask_padding_W*2, mask_padding_H*2,))
    portrai_left, portrai_top, \
    portrai_width, portrai_height = portrait_coordinates
    portrait_mask, portrait_image, portrait_kpts = portrait_images
    portrait_image.save('logs/portrait.png')
    portrait_mask.save('logs/portrait_mask.png')
    portrait_kpts.save('logs/portrait_kpts.png')

    # 
    if enhance_face_region:
        x1, y1, x2, y2 = get_bbox_from_mask(portrait_mask)
        control_mask = np.zeros([portrai_height, portrai_width, 3])
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None
                    
    # Load pipeline
    print(f"\nLoading ControlNet @ {controlnet_path} ...")
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)

    print(f"\nLoading SD-XL Instant-Id Pipeline @ {sdxl_ckpt_path} ...")
    if sdxl_ckpt_path.endswith(".pt") \
    or sdxl_ckpt_path.endswith(".safetensors"):

        pipe = SdXlInstantIdPipeline.from_single_file(
            sdxl_ckpt_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True if sdxl_ckpt_path.endswith(".safetensors") else False,
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

    print(f"\nLoading Instant-Id IP-Adapter @ {face_adapter_path} with strength = {ip_adapter_strength_ratio} ...")
    pipe.load_ip_adapter_instantid(face_adapter_path)
    pipe.set_ip_adapter_scale(ip_adapter_strength_ratio)

    # save VRAM
    print("\nEnabling GPU Efficient Memory ...")
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # load and disable LCM LoRA
    if enable_LCM:
        print("\nEnabling LoRA ...")
        if lora_ckpt_path.endswith('.safetensors'):
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
    
    if prompt is None:
        prompt = "a person"
    
    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
    print(f"\tPositive Prompt: {prompt}, \n\tNegative Prompt: {negative_prompt}")

    # Inference
    print("\nGenerating new face ...")
    generated_face = pipe(
                     image_embeds = face_emb,
                           height = portrai_height,
                            width = portrai_width,
                            image = portrait_image,
                       mask_image = portrait_mask,
                    control_image = portrait_kpts,
                     control_mask = control_mask,
                         strength = mask_strength,

                           prompt = prompt,
                  negative_prompt = negative_prompt,
    controlnet_conditioning_scale = float(identitynet_strength_ratio),
                 ip_adapter_scale = float(ip_adapter_strength_ratio),
              num_inference_steps = int(math.ceil((num_steps / mask_strength))),
                   guidance_scale = guidance_scale,
                        generator = torch.Generator(device=device).manual_seed(seed),
    ).images[0]

    # Clean
    del pipe, controlnet
    if str(device).__contains__("cuda"):
        torch.cuda.empty_cache()

    # Paste the generated face into pose-image
    print("\nReplacing the face in pose-image ...")
    resized_mask = portrait_mask.resize((portrai_width, portrai_height))
    resized_face = generated_face.resize((portrai_width, portrai_height), resample=Image.LANCZOS)
    pose_image.paste(resized_face, (portrai_left, portrai_top), mask=resized_mask)

    return pose_image



    