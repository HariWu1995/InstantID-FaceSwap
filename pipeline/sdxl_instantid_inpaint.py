# based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py
import psutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import PIL.Image

import math
import numpy as np

import torch
import torch.nn.functional as F

from diffusers import StableDiffusionXLControlNetInpaintPipeline as SdXLControlNetInpaintPipeline
from diffusers.models import ControlNetModel
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput as SdXLPipelineOutput
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from ip_adapter.attention_processor import region_control


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. 
    See Section 3.4 of 
        [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


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

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


class StableDiffusionXLInstantIDInpaintPipeline(SdXLControlNetInpaintPipeline):

    def load_ip_adapter_instantid(self, model_ckpt, image_emb_dim=512, num_tokens=16, scale=0.5):
        self.set_image_proj_model(model_ckpt, image_emb_dim, num_tokens)
        self.set_ip_adapter(model_ckpt, num_tokens, scale)

    def set_image_proj_model(self, model_ckpt, image_emb_dim=512, num_tokens=16):

        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.unet.config.cross_attention_dim,
            ff_mult=4,
        )

        image_proj_model.eval()

        self.image_proj_model = image_proj_model.to(self.device, dtype=self.dtype)
        state_dict = torch.load(model_ckpt, map_location="cpu")
        if 'image_proj' in state_dict:
            state_dict = state_dict["image_proj"]
        self.image_proj_model.load_state_dict(state_dict)

        self.image_proj_model_in_features = image_emb_dim

    def set_ip_adapter(self, model_ckpt, num_tokens, scale):

        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size,
                                                   cross_attention_dim=cross_attention_dim,
                                                   scale=scale,
                                                   num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)
        unet.set_attn_processor(attn_procs)

        state_dict = torch.load(model_ckpt, map_location="cpu")
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        if 'ip_adapter' in state_dict:
            state_dict = state_dict['ip_adapter']
        ip_layers.load_state_dict(state_dict)

    def set_ip_adapter_scale(self, scale):
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        for key, value in unet.attn_processors.items():
            cross_attention_dim = None if key.endswith("attn1.processor") else unet.config.cross_attention_dim
            if cross_attention_dim is not None:
                value.scale = scale

    def _encode_prompt_image_emb(self, prompt_image_emb, device, num_images_per_prompt, dtype, do_classifier_free_guidance):

        if isinstance(prompt_image_emb, torch.Tensor):
            prompt_image_emb = prompt_image_emb.clone().detach()
        else:
            prompt_image_emb = torch.tensor(prompt_image_emb)

        self.image_proj_model.to(device)
        prompt_image_emb = prompt_image_emb.to(device=device, dtype=dtype)
        prompt_image_emb = prompt_image_emb.reshape([1, -1, self.image_proj_model_in_features])
        if do_classifier_free_guidance:
            prompt_image_emb = torch.cat([torch.zeros_like(prompt_image_emb), prompt_image_emb], dim=0)
        else:
            prompt_image_emb = torch.cat([prompt_image_emb], dim=0)

        prompt_image_emb = self.image_proj_model(prompt_image_emb)

        bs_embed, seq_len, _ = prompt_image_emb.shape
        prompt_image_emb = prompt_image_emb.repeat(1, num_images_per_prompt, 1)
        prompt_image_emb = prompt_image_emb.view(bs_embed * num_images_per_prompt, seq_len, -1)
        return prompt_image_emb

    @torch.no_grad()
    def __call__(
        self,

        # Inputs
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,

        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        padding_mask_crop = None,

        # Inputs for Inpaint
        control_image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,

        # Embeddings (skip default embedding)
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,

        # Settings
        strength: float = 0.9999,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_rescale: float = 0.0,
        eta: float = 0.0,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
                 aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,

        # Controlnet
        ip_adapter_scale=None,
        control_mask=None,

        # Outputs
        output_type: Optional[str] = "pil",
        return_dict: bool = True,

        # Others
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,

        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        # 0. set ip_adapter_scale
        print('\n\tSetting scale for IP-Adapter ...')
        if ip_adapter_scale is not None:
            self.set_ip_adapter_scale(ip_adapter_scale)

        # 1. Check inputs. Raise error if not correct
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        print('\n\tDefining parameters ...')
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        print('\n\tEncoding prompt ...')
        text_encoder_lora_scale = (
                self.cross_attention_kwargs.get("scale", None) 
             if self.cross_attention_kwargs is not None else None
        )

        (
                        prompt_embeds,
               negative_prompt_embeds,
                 pooled_prompt_embeds,
        negative_pooled_prompt_embeds,  ) = self.encode_prompt(
                                         num_images_per_prompt        =  num_images_per_prompt,
                                                        prompt        =                 prompt,
                                                        prompt_2      =                 prompt_2,
                                               negative_prompt        =        negative_prompt,
                                               negative_prompt_2      =        negative_prompt_2,
                                                        prompt_embeds =                 prompt_embeds,
                                               negative_prompt_embeds =        negative_prompt_embeds,
                                                 pooled_prompt_embeds =          pooled_prompt_embeds,
                                        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                                          do_classifier_free_guidance = self.do_classifier_free_guidance,
                                                            clip_skip = self.clip_skip,
                                                           lora_scale = text_encoder_lora_scale,
                                                               device = device,
                                        )

        # 4. set timesteps
        def denoising_value_valid(dnv):
            return isinstance(denoising_end, float) and 0 < dnv < 1

        print('\n\tSetting timesteps ...')
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
                   num_inference_steps, strength, device, denoising_start=denoising_start 
                                                                       if denoising_value_valid else None
        )

        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )

        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # create a boolean to check if the strength is set to 1. 
        # if so, initialise the latents with pure noise
        is_strength_max = strength == 1.0
        self._num_timesteps = len(timesteps)

        # 5. Preprocess mask and image - resizes image and mask w.r.t height and width
        # 5.1 Prepare init image
        if padding_mask_crop is not None:
            height, width = self.image_processor.get_default_height_width(image, height, width)
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(dtype=torch.float32)

        # 3.2 Encode image prompt
        print('\n\tEncoding image ...')
        prompt_image_emb = self._encode_prompt_image_emb(image_embeds,
                                                         device,
                                                         num_images_per_prompt,
                                                         self.unet.dtype,
                                                         self.do_classifier_free_guidance)

        # 4.1 Region control
        print('\n\tControlling region ...')
        if control_mask is not None:
            mask_weight_image = control_mask
            mask_weight_image = np.array(mask_weight_image)
            mask_weight_image_tensor = torch.from_numpy(mask_weight_image).to(device=device, dtype=prompt_embeds.dtype)
            mask_weight_image_tensor = mask_weight_image_tensor[:, :, 0] / 255.
            mask_weight_image_tensor = mask_weight_image_tensor[None, None]
            h, w = mask_weight_image_tensor.shape[-2:]
            control_mask_wight_image_list = []
            for scale in [8, 8, 8, 16, 16, 16, 32, 32, 32]:
                scale_mask_weight_image_tensor = F.interpolate(
                    mask_weight_image_tensor,(h // scale, w // scale), mode='bilinear')
                control_mask_wight_image_list.append(scale_mask_weight_image_tensor)
            region_mask = torch.from_numpy(np.array(control_mask)[:, :, 0]).to(self.unet.device, dtype=self.unet.dtype) / 255.
            region_control.prompt_image_conditioning = [dict(region_mask=region_mask)]
        else:
            control_mask_wight_image_list = None
            region_control.prompt_image_conditioning = [dict(region_mask=None)]

        # 5.2 Prepare control images
        print('\n\tControlling image ...')
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                #crops_coords=crops_coords,
                #resize_mode=resize_mode,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []
            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    #crops_coords=crops_coords,
                    #resize_mode=resize_mode,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)
                
            control_image = control_images

        else:
            raise ValueError(f"{controlnet.__class__} is not supported.")

        # 5.3 Prepare mask
        print('\n\tPreprocessing mask ...')
        mask = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )

        masked_image = init_image * (mask < 0.5)
        _, _, height, width = init_image.shape

        # 6. Prepare latent variables
        print('\n\tPreparing latents ...')
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        add_noise = True if denoising_start is None else False
        latents_outputs = self.prepare_latents(
            num_images_per_prompt * batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            add_noise=add_noise,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        print('\n\tPreparing mask ...')
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )

        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # 8.1 Prepare extra step kwargs.
        print('\n\tPreparing extra ...')
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8.2 Create tensor stating which controlnets to keep
        print('\n\tCreating tensor for ControlNet ...')
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            if isinstance(self.controlnet, MultiControlNetModel):
                controlnet_keep.append(keeps)
            else:
                controlnet_keep.append(keeps[0])

        # 9. Prepare extra step kwargs. 
        # TODO: Logic should ideally just be moved out of the pipeline
        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 10. Prepare added time ids & embeddings
        print('\n\tPreparing time ids & embeddings ...')
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)
        encoder_hidden_states = torch.cat([prompt_embeds, prompt_image_emb], dim=1)

        # 11. Denoising loop
        print('\n\tDenoising ...')
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if (
                denoising_end is not None
            and denoising_start is not None
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(denoising_start)
            and denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {denoising_start} cannot be larger than or equal to "
              + f"`denoising_end`:  {denoising_end} when using type float."
            )

        elif denoising_end is not None and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds, 
                       "time_ids": add_time_ids,
                }

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                           "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                # Resize control_image to match the size of the input to the controlnet
                # if control_image.shape[-2:] != control_model_input.shape[-2:]:
                #     control_image = F.interpolate(control_image, size=control_model_input.shape[-2:], mode="bilinear", align_corners=False)

                down_block_res_samples, \
                 mid_block_res_sample = controlnet(
                                            control_model_input,
                                            t,
                                            encoder_hidden_states=prompt_image_emb,
                                            controlnet_cond=control_image,
                                            conditioning_scale=cond_scale,
                                            guess_mode=guess_mode,
                                            added_cond_kwargs=controlnet_added_cond_kwargs,
                                            return_dict=False,
                                        )

                # controlnet mask
                if control_mask_wight_image_list is not None:
                    down_block_res_samples = [
                        down_block_res_sample * mask_weight
                    for down_block_res_sample, mask_weight in zip(down_block_res_samples, control_mask_wight_image_list)
                    ]
                    mid_block_res_sample *= control_mask_wight_image_list[-1]

                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                # Callback(s)
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Format output
        if not output_type == "latent":

            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                print('\n\tUpcasting ...')
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            # unscale / denormalize the latents
            # denormalize with the mean and std if available and not None
            print('\n\tDe-normalizing ...')
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            print('\n\tDecoding ...')
            out_image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16, if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            out_image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                out_image = self.watermark.apply_watermark(out_image)

            print('\n\tPostprocessing ...')
            out_image = self.image_processor.postprocess(out_image, output_type=output_type)

        # Offload all parameters
        if kwargs.get('offload_to_cpu', False):
            print('\n\tOffloading all parameters ...')
            self.maybe_free_model_hooks()

        print('\n\tReturning outputs ...')
        if not return_dict:
            return (out_image,)
        return SdXLPipelineOutput(images=out_image)

