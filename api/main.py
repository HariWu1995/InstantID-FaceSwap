import sys
sys.path.append('./')

import os
import random
import argparse
import traceback

import numpy as np
import PIL
from PIL import Image
from io import BytesIO
from typing import Union, Literal, List, Tuple

import uvicorn

from fastapi import FastAPI, Form, File, UploadFile, Request, status
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse, Response
from fastapi.exceptions import HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware

from api.templates import OutputAPI
from api.conversion import image2base64
from api.io_util import load_config, load_multipart_file
from api.face_util import generate_image, STYLE_NAMES, STYLE_DEFAULT


STYLE_NAMES = tuple(STYLE_NAMES)


# Model Config
MODEL_CONFIG = dict(
    face_encoder_dir = './',
    face_adapter_path = './checkpoints/ip-adapter.bin',
    controlnet_path = './checkpoints/ControlNetModel',
    sdxl_ckpt_path = "wangqixun/YamerMIX_v8",
    lora_ckpt_path = "latent-consistency/lcm-lora-sdxl",
)


# API
API_CONFIG = load_config(path='./api/config.yaml')
API_RESPONDER = OutputAPI()

app = FastAPI(
      root_path =  os.getenv("ROOT_PATH"), 
          title = API_CONFIG['DESCRIPTION']['title'],
    description = API_CONFIG['DESCRIPTION']['overview'],
   openapi_tags = API_CONFIG['TAGS'],
        version = API_CONFIG['VERSION'],
       docs_url = None, 
      redoc_url = None,
)

app.add_middleware(
    CORSMiddleware,         # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'],    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'],    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'],    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    query_params = str(request.query_params)
    openapi_url = app.root_path + app.openapi_url + "?" + query_params
    return get_swagger_ui_html(
        openapi_url = openapi_url,
                title = "Hari Yu - Demo",
    swagger_favicon_url = "https://github.com/HariWu1995/InstantID-FaceSwap/blob/main/assets/favicon.ico",
    )


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.post("/config")
async def config(
    face_encoder_dir  : str = Form(description=API_CONFIG['PARAMETERS']['face_encoder_dir'], default=MODEL_CONFIG['face_encoder_dir']), 
    face_adapter_path : str = Form(description=API_CONFIG['PARAMETERS']['face_adapter_path'], default=MODEL_CONFIG['face_adapter_path']), 
    controlnet_path   : str = Form(description=API_CONFIG['PARAMETERS']['controlnet_path'], default=MODEL_CONFIG['controlnet_path']), 
    sdxl_ckpt_path    : str = Form(description=API_CONFIG['PARAMETERS']['sdxl_ckpt_path'], default=MODEL_CONFIG['sdxl_ckpt_path']), 
    lora_ckpt_path    : str = Form(description=API_CONFIG['PARAMETERS']['lora_ckpt_path'], default=MODEL_CONFIG['lora_ckpt_path']), 
):
    try:
        global MODEL_CONFIG
        MODEL_CONFIG.update(dict(
            face_encoder_dir = face_encoder_dir,
            face_adapter_path = face_adapter_path,
            controlnet_path = controlnet_path,
            sdxl_ckpt_path = sdxl_ckpt_path,
            lora_ckpt_path = lora_ckpt_path,
        ))
        response = API_RESPONDER.result(is_successful=True, data=MODEL_CONFIG)

    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())
    
    return response


@app.post("/generate", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def generate(
         face_image: UploadFile = \
                           File(description=API_CONFIG['PARAMETERS']['face_image'], media_type='multipart/form-data'),
         style_name: Literal[STYLE_NAMES] = \
                           Form(description=API_CONFIG['PARAMETERS']['style_name'], default=STYLE_DEFAULT),
             prompt: str = Form(description=API_CONFIG['PARAMETERS']['prompt_positive'], default='a person'), 
    negative_prompt: str = Form(description=API_CONFIG['PARAMETERS']['prompt_negative'], default=''), 
          num_steps: int = Form(description=API_CONFIG['PARAMETERS']['num_steps'], default=5), 
     guidance_scale: int = Form(description=API_CONFIG['PARAMETERS']['guidance_scale'], default=0), 
               seed: int = Form(description=API_CONFIG['PARAMETERS']['seed'], default=3_3_2023), 
         enable_LCM: bool = Form(description=API_CONFIG['PARAMETERS']['enable_LCM'], default=True), 
       enhance_face: bool = Form(description=API_CONFIG['PARAMETERS']['enhance_face'], default=True),
    strength_ip_adapter: float = Form(description=API_CONFIG['PARAMETERS']['strength_ip_adapter'], default=0.8), 
    strength_identitynet: float = Form(description=API_CONFIG['PARAMETERS']['strength_identitynet'], default=0.8), 
):

    try:        
        # Preprocess
        filename = face_image.filename
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in [".png", ".jpg", ".jpeg"]:
            raise TypeError(f"{filename} with type")

        face_image = await face_image.read()
        face_image = Image.open(BytesIO(face_image)).convert('RGB')

        # Run pipeline
        generated_image = generate_image(
                                face_image = face_image, 
                                pose_image = None, 
                                    prompt = prompt, 
                           negative_prompt = negative_prompt, 
                                style_name = style_name, 
                                 num_steps = num_steps, 
                identitynet_strength_ratio = strength_identitynet, 
                    adapter_strength_ratio = strength_ip_adapter, 
                            guidance_scale = guidance_scale, 
                                      seed = seed, 
                                enable_LCM = enable_LCM, 
                       enhance_face_region = enhance_face, 
                              MODEL_CONFIG = MODEL_CONFIG,
        )

        # Response
        print('\nResponding ...')
        if isinstance(generated_image, np.ndarray):
            image_in_bytes = generated_image.tobytes()
        elif isinstance(generated_image, PIL.Image):
            image_in_bytes = BytesIO()
            generated_image.save(image_in_bytes, format='PNG')
            image_in_bytes = image_in_bytes.getvalue()
        else:
            raise TypeError(f"Type of output = {generated_image.__class__} is not supported!")

        response = Response(content=image_in_bytes, media_type="image/png")
        # response = API_RESPONDER.result(is_successful=True, data=results)

    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())

    return response


@app.post("/faceswap", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def faceswap(
         face_image: UploadFile = \
                           File(description=API_CONFIG['PARAMETERS']['face_image'], media_type='multipart/form-data'),
         pose_image: UploadFile = \
                           File(description=API_CONFIG['PARAMETERS']['pose_image'], media_type='multipart/form-data'),
         style_name: Literal[STYLE_NAMES] = \
                           Form(description=API_CONFIG['PARAMETERS']['style_name'], default=STYLE_DEFAULT),
             prompt: str = Form(description=API_CONFIG['PARAMETERS']['prompt_positive'], default='a person'), 
    negative_prompt: str = Form(description=API_CONFIG['PARAMETERS']['prompt_negative'], default=''), 
          num_steps: int = Form(description=API_CONFIG['PARAMETERS']['num_steps'], default=5), 
     guidance_scale: int = Form(description=API_CONFIG['PARAMETERS']['guidance_scale'], default=0), 
               seed: int = Form(description=API_CONFIG['PARAMETERS']['seed'], default=3_3_2023), 
         enable_LCM: bool = Form(description=API_CONFIG['PARAMETERS']['enable_LCM'], default=True), 
       enhance_face: bool = Form(description=API_CONFIG['PARAMETERS']['enhance_face'], default=True),
    strength_ip_adapter: float = Form(description=API_CONFIG['PARAMETERS']['strength_ip_adapter'], default=0.8), 
    strength_identitynet: float = Form(description=API_CONFIG['PARAMETERS']['strength_identitynet'], default=0.8), 
):

    try:        
        # Preprocess
        filename = face_image.filename
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in [".png", ".jpg", ".jpeg"]:
            raise TypeError(f"{filename} with type")

        face_image = await face_image.read()
        face_image = Image.open(BytesIO(face_image)).convert('RGB')

        pose_image = await pose_image.read()
        pose_image = Image.open(BytesIO(pose_image)).convert('RGB')

        # Run pipeline
        generated_image = generate_image(
                                face_image = face_image, 
                                pose_image = pose_image, 
                                    prompt = prompt, 
                           negative_prompt = negative_prompt, 
                                style_name = style_name, 
                                 num_steps = num_steps, 
                identitynet_strength_ratio = strength_identitynet, 
                    adapter_strength_ratio = strength_ip_adapter, 
                            guidance_scale = guidance_scale, 
                                      seed = seed, 
                                enable_LCM = enable_LCM, 
                       enhance_face_region = enhance_face, 
                              MODEL_CONFIG = MODEL_CONFIG,
        )

        # Response
        print('\nResponding ...')
        if isinstance(generated_image, np.ndarray):
            image_in_bytes = generated_image.tobytes()
        elif isinstance(generated_image, PIL.Image):
            image_in_bytes = BytesIO()
            generated_image.save(image_in_bytes, format='PNG')
            image_in_bytes = image_in_bytes.getvalue()
        else:
            raise TypeError(f"Type of output = {generated_image.__class__} is not supported!")

        response = Response(content=image_in_bytes, media_type="image/png")
        # response = API_RESPONDER.result(is_successful=True, data=results)

    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())

    return response


if __name__ == "__main__":

    # Run application
    uvicorn.run(app, **API_CONFIG['SERVER'])

