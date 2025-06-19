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
from easyanimate.utils.utils import save_videos_grid, get_image_to_video_latent
import argparse

import openai
import json
import time
from pathlib import Path
import io
import base64
import requests
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import yaml
# import anthropic
LLM_config_yaml = yaml.load(open('./LLM_CONFIG/llm_config.yaml', 'r'), Loader=yaml.SafeLoader)
APIKEY = LLM_config_yaml['APIKEY']
API_BASE = LLM_config_yaml['API_BASE']
API_MODEL_NAME = LLM_config_yaml['API_MODEL_NAME']

INSTRUCTION_PROMPT = """
## List of Possible Dynamic Objects
{inpainting_entities}
## Instruction
First stage (identify): You should check the objects in the List of Possible Dynamic Objects one by one to verify if they exist in the given image. If there are no dynamic objects in the list indentified in the given image, then you need to indentify some by yourself. 
Second stage (describe): For each indentified object, provide a concise desciption of the Visual Significance (i.e. the proportion in given image), Motion Possibility (i.e. possiblity to contain strong motion in the next few seconds) and what motion it/they may have in the image. 
Third stage (write): Write a dynamical description for the indentified objects, first describe those with strong visual significance and motion possibility.
## Output Format Example (Specify the output format, note that the dynamical description style should be similar to the given example, i.e. one sentence for each object and be concise)
```json
{{  "Think Log": "First stage: 1. Dog is not identified in the given image. 2. Roses are identified in the given image. 3. Flamingos are not identified in the given image. 4. Stream is indentified in the given image. Second Stage: 1. Roses taks a large proportion of the image. Roses seems swinging in the image. (Visual Significance: High, Motion Possibility: Low). 2. Stream is in background and just take a small portion of the given image, stream seems flowing in the image. (Visual Significance: Low, Motion Possibility: High). Third stage: The visual summary for dynamical description is 'The roses swing. The stream flows.' " , 
    "Dynamical Description": "The roses swing. The stream flows."
}}
```
"""


openai.api_key = APIKEY
openai.api_base = API_BASE

# Low gpu memory mode, this is used when the GPU memory is under 16GB
low_gpu_memory_mode = False

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
video_length        = 48
fps                 = 24

# If you want to generate ultra long videos, please set partial_video_length as the length of each sub video segment
partial_video_length = None
overlap_video_length = 4

weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
# validation_image_start  = "asset/0.png"
validation_image_end    = None #"asset/6.png"

# prompts
# prompt                  = "A fast camera flyover, cruising steadily across the scene. Clean motion, flowing without jitters, moving fluidly through the air."
# prompt                  = "camera flythrough shot, smooth motion"
# Strong camera movement. Strong camera motion. Strong Camera zoom.
negative_prompt         = "The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion. "
# negative_prompt         = "Camera movement. Camera motion. Camera zoom. The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion. "
guidance_scale          = 8.5 #M
seed                    = 43
num_inference_steps     = 25
lora_weight             = 0.55
save_path               = "samples/easyanimate-videos_i2v"

config = OmegaConf.load(config_path)

def run_inference(prompt, validation_image_start, savedir_for_wonderjourney, GPT4o_response, inpainting_entities):
    global video_length
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

    if low_gpu_memory_mode:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, "cuda")

    if partial_video_length is not None:
        init_frames = 0
        # last_frames = init_frames + partial_video_length
        # while init_frames < video_length:
        #     if last_frames >= video_length:
        #         if pipeline.vae.quant_conv.weight.ndim==5:
        #             mini_batch_encoder = pipeline.vae.mini_batch_encoder
        #             _partial_video_length = video_length - init_frames
        #             _partial_video_length = int(_partial_video_length // mini_batch_encoder * mini_batch_encoder)
        #         else:
        #             _partial_video_length = video_length - init_frames

        #         if _partial_video_length <= 0:
        #             break
        #     else:
        #         _partial_video_length = partial_video_length

        #     input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image, None, video_length=_partial_video_length, sample_size=sample_size)

        #     with torch.no_grad():
        #         sample = pipeline(
        #             prompt = prompt + ". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic. ", 
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

        #     if last_frames >= video_length:
        #         break

        #     validation_image = [
        #         Image.fromarray(
        #             (sample[0, :, _index].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
        #         ) for _index in range(-overlap_video_length, 0)
        #     ]

        #     init_frames = init_frames + _partial_video_length - overlap_video_length
        #     last_frames = init_frames + _partial_video_length
    else:
        video_length = int(video_length // vae.mini_batch_encoder * vae.mini_batch_encoder) if video_length != 1 else 1
        input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)

        with torch.no_grad():
            sample = pipeline(
                0,
                0.00,
                "Interp_Origin",
                prompt = prompt, 
                video_length = video_length,
                negative_prompt = negative_prompt,
                height      = sample_size[0],
                width       = sample_size[1],
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,

                video        = input_video,
                mask_video   = input_video_mask,
                clip_image   = clip_image, 
            ).videos

    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, "cuda")

    if savedir_for_wonderjourney is None: #M
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        index = len([path for path in os.listdir(save_path)]) + 1
        prefix = str(index).zfill(8)
        video_path = os.path.join(save_path, prefix + ".mp4")
    else:
        video_path = savedir_for_wonderjourney
        
        if GPT4o_response is not None:
            # 目标文件路径
            target_path = savedir_for_wonderjourney + '/GPT4o_response.txt'
            # 将字符串写入文件
            with open(target_path, 'w', encoding='utf-8') as file:
                file.write(GPT4o_response)        
                if inpainting_entities is not None:
                    file.write("inpainting_entities: " + inpainting_entities)   
                
    save_videos_grid(sample, video_path, fps=fps)

def encode_image( image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')  

def generate_text_prompt( image_path, inpainting_entities):

    if inpainting_entities is None:
        print("USE FIXED generated_prompt")
        generated_prompt = "The video is of high quality, high dynamic, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic. "
        # generated_prompt = "Fixed Camera, Strong Motion, Strong dynamic, High dynamic. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic. "
        # generated_prompt = "Cinemagraph, Strong dynamic, High dynamic, Fixed Camera, Static Shot. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic. "
        # Time Lapse
        return generated_prompt, None
    
    else:
        inpainting_entities = str(inpainting_entities.split(", "))
        print("USE LLM to generate generated_prompt")
        print("inpainting_entities:"+ inpainting_entities)

        base64_image = encode_image(image_path)

        if API_BASE.endswith('anthropic'):
            client = anthropic.Client(api_key=APIKEY, base_url=API_BASE)
            # Use Anthropic API
            for i in range(3):
                try:
                    message = client.messages.create(
                        model=API_MODEL_NAME,
                        max_tokens=4096,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text", 
                                        "text": INSTRUCTION_PROMPT.format(inpainting_entities=str(inpainting_entities))
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": base64_image
                                        }
                                    }
                                ]
                            }
                        ]
                    )
                    print("*#*#*#*#*#*#*# FULL Response: *#*#*#*#*#*#*#")
                    print(message)
                    print("*#*#*#*#*#*#*# MLLM Response: *#*#*#*#*#*#*#")
                    GPT4o_response = message.content[0].text
                    print(GPT4o_response)
                    generated_prompt = read_LLM_Completion(GPT4o_response)["Dynamical Description"] + "The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic. "

                    return generated_prompt, str(GPT4o_response)
                except Exception as e:
                    print(e)
                    time.sleep(1)
                    continue
        else:
            # Use original OpenAI API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }

            payload = {
                "model": API_MODEL_NAME,
                "messages": [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": ""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                    ]
                }
                ],
                "max_tokens": 4096
            }
            payload['messages'][0]['content'][0]['text'] = INSTRUCTION_PROMPT.format(
                    inpainting_entities = str(inpainting_entities)
                )
            for i in range(3):
                try:
                    response = requests.post(API_BASE + "/chat/completions", headers=headers, json=payload, timeout = 60)
                    print("*#*#*#*#*#*#*# FULL Response: *#*#*#*#*#*#*#")
                    print(response)
                    print("*#*#*#*#*#*#*# MLLM Response: *#*#*#*#*#*#*#")
                    GPT4o_response = response.json()['choices'][0]['message']['content']
                    print(GPT4o_response)
                    generated_prompt = read_LLM_Completion(response.json()['choices'][0]['message']['content'])["Dynamical Description"] + "The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic. "

                    return generated_prompt, str(GPT4o_response)
                except Exception as e:
                    print(e)
                    time.sleep(1)
                    continue

    

def read_LLM_Completion(text):
    import re

    pattern = r"(?:.*?```json)(.*?)(?:```.*?)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return json.loads(match.group(1).strip())

    pattern = r"\{.*\}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0).strip())
        except Exception:
            pass
    raise ("bad format!")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', required= False)  
    parser.add_argument('--image_path', required= True)  
    parser.add_argument('--inpainting_entities', required= False)  
    
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    text_Prompt, GPT4o_response = generate_text_prompt(args.image_path, args.inpainting_entities)
    # Run inference
    run_inference(text_Prompt, args.image_path, args.savedir, GPT4o_response, args.inpainting_entities)


