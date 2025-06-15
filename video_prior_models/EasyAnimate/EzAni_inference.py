import os
import numpy as np
import torch
import cv2
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection,  CLIPImageProcessor

from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer3d import Transformer3DModel
from easyanimate.models.transformer3d import Transformer3DModel, HunyuanTransformer3DModel
from easyanimate.pipeline.pipeline_easyanimate_multi_text_encoder_inpaint import EasyAnimatePipeline_Multi_Text_Encoder_Inpaint
from easyanimate.pipeline.pipeline_easyanimate_inpaint import \
    EasyAnimateInpaintPipeline
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import save_videos_grid, get_image_to_video_latent, create_video_from_images, get_priorImages_to_video_latent

import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

import torch

from pathlib import Path
import pickle
import shutil
from torchvision.io import write_video
import imageio
import numpy as np
import cv2
import PIL
from PIL import Image
import os
import json
import math
import datetime
import time
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import os
import cv2
import numpy as np
import PIL.Image 

import subprocess
import os

# result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
# output = result.stdout
# for line in output.splitlines():
#     if '=' in line:
#         var, value = line.split('=', 1)
#         os.environ[var] = value

#------- Mode for spaceDyn -------------
MODE = "Interp_PriorPS"

#------- General Setting -----------
# GPU_MODE = "low"
GPU_MODE = "defualt"
# GPU_MODE = "high"


interpolation_mode = MODE 
VISUALIZE = True
SWAP_VALIDATION_IMAGE = False
TEMPERAL_PADDING = 4 # Also check TEMPERAL_LATENT_PADDING in easyanimate/pipeline/pipeline_easyanimate_inpaint.py
TEMPERAL_PAD_CUT_MODE = "Direct Cut"

#------- Hyper Pamameters ------------
INFER_STEP = 25
MASK_LATENT_CUT = 0.8
KERNEL_SIZE = 1
TEMPERAL_KERNAL = 1 # must be odd
PRIOR_STOP = 20 #PPP
PREPROCESS_CV2_TELEA = True


#------- Global Setting by EasyAnimate ------------
# Low gpu memory mode, this is used when the GPU memory is under 16GB
# low_gpu_memory_mode = False

# Config and model path
config_path         = "./video_prior_models/EasyAnimate/config/easyanimate_video_slicevae_motion_module_v3.yaml"
model_name          = "./video_prior_models/EasyAnimate/models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
sampler_name        = "Euler"

# Load pretrained model if need
transformer_path    = None
# V2 and V3 does not need a motion module
motion_module_path  = None
vae_path            = None
lora_path           = None

# Other params
sample_size         = [512, 512]
# In EasyAnimateV1, the video_length of video is 40 ~ 80.
# In EasyAnimateV2 and V3, the video_length of video is 1 ~ 144. If u want to generate a image, please set the video_length = 1.
FRAME_MODE        = 48 
FRAME_MODE        = FRAME_MODE + TEMPERAL_PADDING * 2
fps               = 24

# If you want to generate ultra long videos, please set partial_video_length as the length of each sub video segment
partial_video_length = None
overlap_video_length = 4

weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
# validation_image_start  = "asset/0.png"
# validation_image_end    = "asset/6.png"

# prompts
# prompt                  = "A fast camera flyover, cruising steadily across the scene. Clean motion, flowing without jitters, moving fluidly through the air."
# prompt                  = "camera flythrough shot, smooth motion"
negative_prompt         = "The video has blurry area. The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion. "
guidance_scale          = 7
seed                    = 439
num_inference_steps     = INFER_STEP
# num_inference_steps     = 100 
lora_weight             = 0.55
# save_path               = "samples/easyanimate-videos_i2v"

config = OmegaConf.load(config_path)

def run_inference(prior_stop, lr, prompt, masks, prior_images, prior_mask_latents, validation_image_start, validation_image_end, output_frame_dir, result_analyze_dir, visualize_folder, interpolation_mode): 
    global FRAME_MODE

    # Get Transformer
    if config['enable_multi_text_encoder']:
        Choosen_Transformer3DModel = HunyuanTransformer3DModel
    else:
        Choosen_Transformer3DModel = Transformer3DModel

    transformer = Choosen_Transformer3DModel.from_pretrained_2d(
        model_name, 
        subfolder="transformer",
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs'])
    ).to(weight_dtype)

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if motion_module_path is not None:
        print(f"From Motion Module: {motion_module_path}")
        if motion_module_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(motion_module_path)
        else:
            state_dict = torch.load(motion_module_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}, {u}")

    # Get Vae
    if OmegaConf.to_container(config['vae_kwargs'])['enable_magvit']:
        Choosen_AutoencoderKL = AutoencoderKLMagvit
    else:
        Choosen_AutoencoderKL = AutoencoderKL
    vae = Choosen_AutoencoderKL.from_pretrained(
        model_name, 
        subfolder="vae", 
    ).to(weight_dtype)

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_name, subfolder="image_encoder"
    ).to("cuda", weight_dtype)
    clip_image_processor = CLIPImageProcessor.from_pretrained(
        model_name, subfolder="image_encoder"
    )

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler, 
        "PNDM": PNDMScheduler,
        "DDIM": DDIMScheduler,
    }[sampler_name]

    if config['enable_multi_text_encoder']:
        scheduler = Choosen_Scheduler.from_pretrained(
            model_name, 
            subfolder="scheduler"
        )
        pipeline = EasyAnimatePipeline_Multi_Text_Encoder_Inpaint.from_pretrained(
            model_name,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=weight_dtype,
            clip_image_encoder=clip_image_encoder,
            clip_image_processor=clip_image_processor,
        )
    else:
        scheduler = Choosen_Scheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))

        pipeline = EasyAnimateInpaintPipeline.from_pretrained(
            model_name,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=weight_dtype,
            clip_image_encoder=clip_image_encoder,
            clip_image_processor=clip_image_processor,
        )

    if GPU_MODE == "low":
        pipeline.enable_sequential_cpu_offload()
    elif GPU_MODE == "defualt":
        pipeline.enable_model_cpu_offload()
    elif GPU_MODE == "high":
        print("high gpu mode")
    else:
        NotImplementedError

    generator = torch.Generator(device="cuda").manual_seed(seed)

    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, "cuda")

    if partial_video_length is not None:
        init_frames = 0
        # last_frames = init_frames + partial_video_length
        # while init_frames < FRAME_MODE:
        #     if last_frames >= FRAME_MODE:
        #         if pipeline.vae.quant_conv.weight.ndim==5:
        #             mini_batch_encoder = pipeline.vae.mini_batch_encoder
        #             _partial_video_length = FRAME_MODE - init_frames
        #             _partial_video_length = int(_partial_video_length // mini_batch_encoder * mini_batch_encoder)
        #         else:
        #             _partial_video_length = FRAME_MODE - init_frames

        #         if _partial_video_length <= 0:
        #             break
        #     else:
        #         _partial_video_length = partial_video_length

        #     input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image, None, video_length=_partial_video_length, sample_size=sample_size)

        #     with torch.no_grad():
        #         sample = pipeline(
        #             prompt + ". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic. ", 
        #             video_length = _partial_video_length,
        #             negative_prompt = negative_prompt,
        #             height      = sample_size[0],
        #             width       = sample_size[1],
        #             generator   = generator,
        #             guidance_scale = guidance_scale,
        #             num_inference_steps = num_inference_steps,

        #             video        = input_video,
        #             mask_video   = input_video_mask,
        #             clip_image   = clip_image, 
        #         ).videos

        #     if init_frames != 0:
        #         mix_ratio = torch.from_numpy(
        #             np.array([float(_index) / float(overlap_video_length) for _index in range(overlap_video_length)], np.float32)
        #         ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        #         new_sample[:, :, -overlap_video_length:] = new_sample[:, :, -overlap_video_length:] * (1 - mix_ratio) + \
        #             sample[:, :, :overlap_video_length] * mix_ratio
        #         new_sample = torch.cat([new_sample, sample[:, :, overlap_video_length:]], dim = 2)

        #         sample = new_sample
        #     else:
        #         new_sample = sample

        #     if last_frames >= FRAME_MODE:
        #         break

        #     validation_image = [
        #         Image.fromarray(
        #             (sample[0, :, _index].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
        #         ) for _index in range(-overlap_video_length, 0)
        #     ]

        #     init_frames = init_frames + _partial_video_length - overlap_video_length
        #     last_frames = init_frames + _partial_video_length
    else:
        FRAME_MODE = int(FRAME_MODE // vae.mini_batch_encoder * vae.mini_batch_encoder) if FRAME_MODE != 1 else 1
        input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=FRAME_MODE, sample_size=sample_size)
        prior_video = get_priorImages_to_video_latent(prior_images) #M
        if True: #for Flip
            input_video_flip, input_video_mask_flip, clip_image_flip = get_image_to_video_latent(validation_image_end, validation_image_start, video_length=FRAME_MODE, sample_size=sample_size)
            prior_video_flip = get_priorImages_to_video_latent(prior_images[::-1]) #M
            masks_flip =  torch.flip(masks, [2]) # flip the prior masks 
        


        with torch.no_grad():
            sample = pipeline(
                prior_stop,
                lr,
                interpolation_mode,
                prompt, 
                visualize_folder,
                video_length = FRAME_MODE,
                negative_prompt = negative_prompt,
                height      = sample_size[0],
                width       = sample_size[1],
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,
                # for forward
                video        = input_video,
                mask_video   = input_video_mask,
                clip_image   = clip_image, 
                prior_video  = prior_video, #M
                prior_mask  = masks, #M
                # for fliped
                video_flip = input_video_flip, 
                mask_video_flip = input_video_mask_flip, 
                clip_image_flip = clip_image_flip,
                prior_video_flip = prior_video_flip, #M flip
                prior_mask_flip  = masks_flip, #M flip

                prior_mask_latents = prior_mask_latents, #M
            ).videos

    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, "cuda")

    os.makedirs(output_frame_dir, exist_ok=True)  
    save_videos_grid(sample, str(output_frame_dir), fps=fps, temperal_pad_cut_mode = TEMPERAL_PAD_CUT_MODE, temperal_pad = TEMPERAL_PADDING)

    if result_analyze_dir is not None:

        save_videos_grid(sample, str(result_analyze_dir / "output_frame"), fps=fps)

        create_video_from_images(str((result_analyze_dir / "input_frame_Masked")), fps=fps)
        create_video_from_images(str((result_analyze_dir / "input_mask")), fps=fps)
        create_video_from_images(str((result_analyze_dir / "origin_frame")), fps=fps)
        create_video_from_images(str((result_analyze_dir / "origin_mask")), fps=fps)

        if TEMPERAL_KERNAL > 0:
            create_video_from_images(str((result_analyze_dir / "depth_dilated_input_frame_Masked")), fps=fps)
            create_video_from_images(str((result_analyze_dir / "depth_dilated_input_mask")), fps=fps)

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path, exist_ok=True)

    # index = len([path for path in os.listdir(save_path)]) + 1
    # prefix = str(index).zfill(8)

    # if FRAME_MODE == 1:
    #     save_sample_path = os.path.join(save_path, prefix + f".png")

    #     image = sample[0, :, 0]
    #     image = image.transpose(0, 1).transpose(1, 2)
    #     image = (image * 255).numpy().astype(np.uint8)
    #     image = Image.fromarray(image)
    #     image.save(save_sample_path)
    # else:
    #     video_path = os.path.join(save_path, prefix + ".mp4")
    #     save_videos_grid(sample, video_path, fps=fps)


# 定义一个仅沿着深度维度进行膨胀的卷积层
class DilateAlongDepth(nn.Module):
    def __init__(self, kernel_size_depth):
        super(DilateAlongDepth, self).__init__()
        self.kernel_size_depth = kernel_size_depth
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size_depth, stride=1, padding=kernel_size_depth//2, bias=False)

        # 初始化卷积核权重
        kernel = torch.ones(1, 1, kernel_size_depth, dtype=torch.float32)  # 明确指定 dtype
        self.conv1d.weight.data.copy_(kernel)

    def forward(self, x):
        # 获取原始输入张量的维度
        batch_size, channels, depth, height, width = x.size()

        # 将输入张量重塑为适合1D卷积的形式
        x = x.permute(0, 1, 3, 4, 2)  # 调整维度顺序为 (batch, channel, height, width, depth)
        x = x.contiguous().view(batch_size * channels * height * width, 1, depth)  # 重塑为 (batch * channel * height * width, 1, depth)

        # 执行卷积操作
        x = self.conv1d(x)

        # 重塑回原来的3D形式
        x = x.contiguous().view(batch_size, channels, height, width, depth)
        x = x.permute(0, 1, 4, 2, 3)  # 调整维度顺序回 (batch, channel, depth, height, width)

        # binarize
        x = (x > 0.5).float()
        return x

def pad_images(frames_dir, masks_dir, frames_padded_dir, masks_padded_dir, N):
    # 确保目标目录存在
    os.makedirs(frames_padded_dir, exist_ok=True)
    os.makedirs(masks_padded_dir, exist_ok=True)

    # 读取所有frame和mask图片
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

    # 检查图片数量是否一致
    if len(frame_files) != len(mask_files):
        raise ValueError("Frame and mask images do not match in number.")

    # 读取第一张和最后一张图片
    first_frame = Image.open(os.path.join(frames_dir, frame_files[0]))
    last_frame = Image.open(os.path.join(frames_dir, frame_files[-1]))
    first_mask = Image.open(os.path.join(masks_dir, mask_files[0]))
    last_mask = Image.open(os.path.join(masks_dir, mask_files[-1]))

    # 保存padding图片
    for i in range(N):
        first_frame.save(os.path.join(frames_padded_dir, f'{i}.png'))
        first_mask.save(os.path.join(masks_padded_dir, f'{i}.png'))

    # 保存原始图片
    for idx, (frame_file, mask_file) in enumerate(zip(frame_files, mask_files)):
        frame = Image.open(os.path.join(frames_dir, frame_file))
        mask = Image.open(os.path.join(masks_dir, mask_file))
        frame.save(os.path.join(frames_padded_dir, f'{idx + N}.png'))
        mask.save(os.path.join(masks_padded_dir, f'{idx + N}.png'))

    # 保存padding图片
    for i in range(N):
        last_frame.save(os.path.join(frames_padded_dir, f'{len(frame_files) + N + i}.png'))
        last_mask.save(os.path.join(masks_padded_dir, f'{len(mask_files) + N + i}.png'))


def cv2_telea(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return ret, mask

def import_warped_image(origin_frame_dir, origin_mask_dir, result_analyze_dir):

    num_frames = FRAME_MODE 

    # Set size  
    H, W = 512, 512
    latent_H, latent_W = 64, 64

    # Import prior frames
    prior_images = []
    masks = []
    for render_id in range(0 , num_frames):
        # read original frames and masks
        if not SWAP_VALIDATION_IMAGE:
            origin_idx = num_frames -1 - render_id
        else:
            origin_idx = render_id
        origin_frame_path = str(origin_frame_dir/ (str(origin_idx)+".png"))
        origin_mask_path = str(origin_mask_dir / (str(origin_idx)+".png"))

        origin_frame_array = np.array( PIL.Image.open(origin_frame_path).convert("RGB") .resize((W,H),PIL.Image.Resampling.NEAREST))
        origin_mask_array = np.array( PIL.Image.open(origin_mask_path).resize((W,H),PIL.Image.Resampling.NEAREST))

        if PREPROCESS_CV2_TELEA:
            # fill in image
            img = origin_frame_array
            fill_mask_ = origin_mask_array
            for _ in range(3):
                img, _ = cv2_telea(img, fill_mask_)
            origin_frame_array = img

        if len(np.unique(origin_mask_array)) > 2:
            raise("origin_mask_array is not binarized. Values in the image can only be 0 or 255.") 
        else:
            origin_mask_array[origin_mask_array == 0] = 0
            origin_mask_array[origin_mask_array == 255] = 1
            origin_mask_array = np.repeat(origin_mask_array[:,:,np.newaxis]*255.,repeats=3,axis=2)

        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        _input_mask = cv2.dilate(np.array(origin_mask_array), kernel, iterations = 1)
        input_mask = PIL.Image.fromarray(np.uint8(_input_mask))
        input_mask_array = np.array(input_mask)

        if len(np.unique(input_mask_array)) > 2:
            raise("input_mask_array is not binarized. Values in the image can only be 0 or 255.") 
        else:
            input_mask_array[input_mask_array == 0] = 0
            input_mask_array[input_mask_array == 255] = 1
            masks.append(torch.from_numpy( np.mean(input_mask_array,axis = -1) ).unsqueeze(0))

        # set input frames (input_frame is used as real input prior image, input_frame_Masked is used for visualize)
        input_frame = PIL.Image.fromarray(np.uint8(origin_frame_array))
        input_frame_Masked = PIL.Image.fromarray(np.uint8(input_frame*(1-input_mask_array)))
        prior_images.append(input_frame)       

        # save images for result analysis
        if result_analyze_dir is not None:
            shutil.copy(origin_frame_path, str(result_analyze_dir / "origin_frame" / (str(render_id)+".png")))
            shutil.copy(origin_mask_path, str(result_analyze_dir / "origin_mask" / (str(render_id)+".png")))
            input_frame_Masked.save(str(result_analyze_dir / "input_frame_Masked" / (str(render_id)+".png")))
            input_mask.save(str(result_analyze_dir / "input_mask" / (str(render_id)+".png")))
    
    masks = torch.cat(masks).unsqueeze(0).unsqueeze(0)

    if TEMPERAL_KERNAL > 0:
        # perform depth dulation
        with torch.no_grad():
            # 1. get depth_dilated_mask
            dilate_depth_layer = DilateAlongDepth(kernel_size_depth=TEMPERAL_KERNAL).to(dtype = masks.dtype)
            depth_dilated_mask = dilate_depth_layer(masks)

            # 2. get corresponding depth_dilated_mask_latents

            # 定义池化核的大小和步幅
            kernel_size = 8
            stride = 8

            # 使用 avg_pool3d 进行平均池化
            averaged_mask_inverted = F.avg_pool3d(1 - depth_dilated_mask, kernel_size=kernel_size, stride=stride)  # [1, 1, 48/8, 512/8, 512/8]
            averaged_mask_inverted_cuted = (averaged_mask_inverted > MASK_LATENT_CUT)
            averaged_mask_cuted = torch.logical_not(averaged_mask_inverted_cuted).float()
            prior_mask_latents = averaged_mask_cuted.repeat_interleave(repeats=2, dim=2)

            # get averaged_mask_cuted_upsampled for visualize
            # 定义每个维度上重复的次数
            # b,c,f,h,w = averaged_mask_cuted.shape

            # 定义每个维度上重复的次数
            repeats_dim2 = torch.tensor([8] * averaged_mask_cuted.size(2))  # 在 dim=2 上重复 8 次
            repeats_dim3 = torch.tensor([8] * averaged_mask_cuted.size(3))  # 在 dim=3 上重复 8 次
            repeats_dim4 = torch.tensor([8] * averaged_mask_cuted.size(4))  # 在 dim=4 上重复 8 次
            # 使用 repeat_interleave 进行扩展
            expanded_tensor = averaged_mask_cuted.repeat_interleave(repeats_dim2, dim=2)
            expanded_tensor = expanded_tensor.repeat_interleave(repeats_dim3, dim=3)
            expanded_tensor = expanded_tensor.repeat_interleave(repeats_dim4, dim=4)
            prior_mask_latents_expandForVis = expanded_tensor


        # save depth_dilated_input_mask and depth_dilated_input_frame_Masked for visualize
        if result_analyze_dir is not None:
            for render_id in range(0 , num_frames):
                # read original frames
                if not SWAP_VALIDATION_IMAGE:
                    origin_idx = num_frames -1 - render_id
                else:
                    origin_idx = render_id
                origin_frame_path = str(origin_frame_dir/ (str(origin_idx)+".png"))
                origin_frame_array = np.array( PIL.Image.open(origin_frame_path).convert("RGB") .resize((W,H),PIL.Image.Resampling.NEAREST))

                if PREPROCESS_CV2_TELEA:
                    origin_mask_path = str(origin_mask_dir / (str(origin_idx)+".png"))
                    origin_mask_array = np.array( PIL.Image.open(origin_mask_path).resize((W,H),PIL.Image.Resampling.NEAREST))

                    # fill in image
                    img = origin_frame_array
                    fill_mask_ = origin_mask_array
                    for _ in range(3):
                        img, _ = cv2_telea(img, fill_mask_)
                    origin_frame_array = img

                # set depth_dilated_input_mask 
                # _depth_dilated_input_mask = depth_dilated_mask[0][0][render_id].numpy() # visualize option 1
                _depth_dilated_input_mask = prior_mask_latents_expandForVis[0][0][render_id].numpy() # visualize option 2
                _depth_dilated_input_mask = np.repeat(_depth_dilated_input_mask[:,:,np.newaxis]*255.,repeats=3,axis=2)
                depth_dilated_input_mask = PIL.Image.fromarray(np.uint8(_depth_dilated_input_mask))
                depth_dilated_input_mask_array = np.array(depth_dilated_input_mask)

                if len(np.unique(depth_dilated_input_mask_array)) > 2:
                    raise("depth_dilated_input_mask_array is not binarized. Values in the image can only be 0 or 255.") 
                else:
                    depth_dilated_input_mask_array[depth_dilated_input_mask_array == 0] = 0
                    depth_dilated_input_mask_array[depth_dilated_input_mask_array == 255] = 1

                # set depth_dilated_input_frame_Masked
                input_frame = PIL.Image.fromarray(np.uint8(origin_frame_array))
                depth_dilated_input_frame_Masked = PIL.Image.fromarray(np.uint8(input_frame*(1-depth_dilated_input_mask_array)))

                # save images for result analysis
                depth_dilated_input_frame_Masked.save(str(result_analyze_dir / "depth_dilated_input_frame_Masked" / (str(render_id)+".png")))
                depth_dilated_input_mask.save(str(result_analyze_dir / "depth_dilated_input_mask" / (str(render_id)+".png")))

        return depth_dilated_mask, prior_images, prior_mask_latents

    return masks, prior_images # never use anymore


def get_parser():
    parser = argparse.ArgumentParser()
    # current has defual value
    parser.add_argument('--text_Prompt', default="A fast camera flyover, cruising steadily across the scene. Clean motion, flowing without jitters, moving fluidly through the air.")  
    # support for custom interpolation
    parser.add_argument('--origin_frame_dir', type=str, required=True, help='Path to the origin frame directory')  
    parser.add_argument('--origin_mask_dir', type=str, required=True, help='Path to the origin mask directory')  
    parser.add_argument('--output_frame_dir', type=str, required=True, help='Path to the output frame directory') 
    parser.add_argument('--result_analyze_dir', type=str, help='Path to the result analyze directory') 
    # lagacy
    parser.add_argument('--lr',type=float, required=True)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Set paths
    output_frame_dir = Path(args.output_frame_dir)

    if args.result_analyze_dir is not None:
        result_analyze_dir = Path(args.result_analyze_dir)
        (result_analyze_dir / "origin_frame").mkdir(parents=True ,exist_ok=True)
        (result_analyze_dir / "origin_mask").mkdir(parents=True ,exist_ok=True)
        (result_analyze_dir / "input_frame_Masked").mkdir(parents=True ,exist_ok=True)
        (result_analyze_dir / "input_mask").mkdir(parents=True ,exist_ok=True)
        (result_analyze_dir / "depth_dilated_input_frame_Masked").mkdir(parents=True ,exist_ok=True)
        (result_analyze_dir / "depth_dilated_input_mask").mkdir(parents=True ,exist_ok=True)
        (result_analyze_dir / "output_frame").mkdir(parents=True ,exist_ok=True)
    else:
        result_analyze_dir = None

    origin_frame_dir = Path(args.origin_frame_dir)
    origin_mask_dir = Path(args.origin_mask_dir)

    if TEMPERAL_PADDING > 0:
        frames_padded_dir = origin_frame_dir.parent / "frames_for_svd_inpainting_padded"
        masks_padded_dir = origin_mask_dir.parent / "masks_for_svd_inpainting_padded"
        pad_images(origin_frame_dir, origin_mask_dir, frames_padded_dir, masks_padded_dir, TEMPERAL_PADDING)
        origin_frame_dir = frames_padded_dir
        origin_mask_dir = masks_padded_dir

    if VISUALIZE and result_analyze_dir is not None:
        visualize_folder = result_analyze_dir / "visualize"
        visualize_folder.mkdir(parents=True ,exist_ok=True)
    else:
        visualize_folder = None
        
    # Set validation image
    validation_image_start = str(origin_frame_dir / (str(FRAME_MODE - 1)+ ".png") )
    validation_image_end = str(origin_frame_dir /  (str(0)  + ".png") )
    if SWAP_VALIDATION_IMAGE:
        temp = validation_image_end
        validation_image_end = validation_image_start
        validation_image_start = temp

    # Preprocess: import dilated masks and prior_images
    masks, prior_images, prior_mask_latents = import_warped_image(origin_frame_dir, origin_mask_dir, result_analyze_dir)

    # Run inference
    run_inference(PRIOR_STOP, args.lr, args.text_Prompt, masks, prior_images, prior_mask_latents, validation_image_start, validation_image_end, output_frame_dir, result_analyze_dir, visualize_folder, interpolation_mode)


