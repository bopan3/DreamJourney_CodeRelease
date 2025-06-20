from PIL import Image
from PIL import ImageFilter
import cv2
import numpy as np
import scipy
import scipy.signal
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import torch
import io

from collections import deque
from torchvision.transforms import ToTensor
import os
import yaml
import shutil
from .general_utils import save_video
from datetime import datetime

import subprocess
script_path_dict = {
    "KeyFrameDynamic_1024_DynCraft": "./script4externalModules/KeyFrameDynamic_1024_DynCraft.sh",
    "svd_inpainting": "./script4externalModules/svd_inpainting.sh",
    "KeyFrameDynamic_1024_svd": "./script4externalModules/KeyFrameDynamic_1024_svd.sh",
    "KeyFrameDynamic_EasyAnimate":"./script4externalModules/KeyFrameDynamic_EasyAnimate.sh",
    "KeyFrameDynamic_EasyAnimate_noGPT4Dyn":"./script4externalModules/KeyFrameDynamic_EasyAnimate_noGPT4Dyn.sh",
}

def find_biggest_connected_inpaint_region(mask):
    H, W = mask.shape
    visited = torch.zeros((H, W), dtype=torch.bool)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # up, right, down, left
    
    def bfs(i, j):
        queue = deque([(i, j)])
        region = []
        
        while queue:
            x, y = queue.popleft()
            if 0 <= x < H and 0 <= y < W and not visited[x, y] and mask[x, y] == 1:
                visited[x, y] = True
                region.append((x, y))
                for dx, dy in directions:
                    queue.append((x + dx, y + dy))
                    
        return region
    
    max_region = []
    
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and not visited[i, j]:
                current_region = bfs(i, j)
                if len(current_region) > len(max_region):
                    max_region = current_region
    
    mask_connected = torch.zeros((H, W)).to(mask.device)
    for x, y in max_region:
        mask_connected[x, y] = 1
    return mask_connected


def edge_pad(img, mask, mode=1):
    if mode == 0:
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res0 = 1 - nmask
        res1 = nmask
        p0 = np.stack(res0.nonzero(), axis=0).transpose()
        p1 = np.stack(res1.nonzero(), axis=0).transpose()
        min_dists, min_dist_idx = cKDTree(p1).query(p0, 1)
        loc = p1[min_dist_idx]
        for (a, b), (c, d) in zip(p0, loc):
            img[a, b] = img[c, d]
    elif mode == 1:
        record = {}
        kernel = [[1] * 3 for _ in range(3)]
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res = scipy.signal.convolve2d(
            nmask, kernel, mode="same", boundary="fill", fillvalue=1
        )
        res[nmask < 1] = 0
        res[res == 9] = 0
        res[res > 0] = 1
        ylst, xlst = res.nonzero()
        queue = [(y, x) for y, x in zip(ylst, xlst)]
        # bfs here
        cnt = res.astype(np.float32)
        acc = img.astype(np.float32)
        step = 1
        h = acc.shape[0]
        w = acc.shape[1]
        offset = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            target = []
            for y, x in queue:
                val = acc[y][x]
                for yo, xo in offset:
                    yn = y + yo
                    xn = x + xo
                    if 0 <= yn < h and 0 <= xn < w and nmask[yn][xn] < 1:
                        if record.get((yn, xn), step) == step:
                            acc[yn][xn] = acc[yn][xn] * cnt[yn][xn] + val
                            cnt[yn][xn] += 1
                            acc[yn][xn] /= cnt[yn][xn]
                            if (yn, xn) not in record:
                                record[(yn, xn)] = step
                                target.append((yn, xn))
            step += 1
            queue = target
        img = acc.astype(np.uint8)
    else:
        nmask = mask.copy()
        ylst, xlst = nmask.nonzero()
        yt, xt = ylst.min(), xlst.min()
        yb, xb = ylst.max(), xlst.max()
        content = img[yt : yb + 1, xt : xb + 1]
        img = np.pad(
            content,
            ((yt, mask.shape[0] - yb - 1), (xt, mask.shape[1] - xb - 1), (0, 0)),
            mode="edge",
        )
    return img, mask


def gaussian_noise(img, mask):
    noise = np.random.randn(mask.shape[0], mask.shape[1], 3)
    noise = (noise + 1) / 2 * 255
    noise = noise.astype(np.uint8)
    nmask = mask.copy()
    nmask[mask > 0] = 1
    img = nmask[:, :, np.newaxis] * img + (1 - nmask[:, :, np.newaxis]) * noise
    return img, mask


def cv2_telea(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return ret, mask


def cv2_ns(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
    return ret, mask


def mean_fill(img, mask):
    avg = img.mean(axis=0).mean(axis=0)
    img[mask < 1] = avg
    return img, mask

def estimate_scale_and_shift(x, y, init_method='identity', optimize_scale=True):
    assert len(x.shape) == 1 and len(y.shape) == 1, "Inputs should be 1D tensors"
    assert x.shape[0] == y.shape[0], "Input tensors should have the same length"

    n = x.shape[0]

    if init_method == 'identity':
        shift_init = 0.
        scale_init = 1.
    elif init_method == 'median':
        shift_init = (torch.median(y) - torch.median(x)).item()
        scale_init = (torch.sum(torch.abs(y - torch.median(y))) / n / (torch.sum(torch.abs(x - torch.median(x))) / n)).item()
    else:
        raise ValueError("init_method should be either 'identity' or 'median'")
    shift = torch.tensor(shift_init).cuda().requires_grad_()
    scale = torch.tensor(scale_init).cuda().requires_grad_()

    # Set optimizer and scheduler
    optimizer = torch.optim.Adam([shift, scale], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    # Optimization loop
    for step in range(1000):  # Set the range to the number of steps you find appropriate
        optimizer.zero_grad()
        if optimize_scale:
            loss = torch.abs((x.detach() + shift) * scale - y.detach()).mean()
        else:
            loss = torch.abs(x.detach() + shift - y.detach()).mean()
        loss.backward()
        if step == 0:
            print(f"Iteration {step + 1}: L1 Loss = {loss.item():.4f}")
        optimizer.step()
        scheduler.step(loss)

        # Early stopping condition if needed
        if step > 20 and scheduler._last_lr[0] < 1e-6:  # You might want to adjust these conditions
            print(f"Iteration {step + 1}: L1 Loss = {loss.item():.4f}")
            break

    if optimize_scale:
        return scale.item(), shift.item()
    else:
        return 1., shift.item()


def save_depth_map(depth_map, file_name, vmin=None, vmax=None, save_clean=False):
    depth_map = np.squeeze(depth_map)
    if depth_map.ndim != 2:
        raise ValueError("Depth map after squeezing must be 2D.")

    dpi = 100  # Adjust this value if necessary
    figsize = (depth_map.shape[1] / dpi, depth_map.shape[0] / dpi)  # Width, Height in inches

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cax = ax.imshow(depth_map, cmap='viridis', vmin=vmin, vmax=vmax)

    if not save_clean:
        # Standard save with labels and color bar
        cbar = fig.colorbar(cax)
        ax.set_title("Depth Map")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
    else:
        # Clean save without labels, color bar, or axis
        plt.axis('off')
        ax.set_aspect('equal', adjustable='box')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')  # Convert to RGB
    # img = img.resize((depth_map.shape[1], depth_map.shape[0]), Image.ANTIALIAS)  # Resize to original dimensions
    img = img.resize((depth_map.shape[1], depth_map.shape[0]), Image.Resampling.LANCZOS)  #M Resize to original dimensions
    
    img.save(file_name, format='png')
    buf.close()
    plt.close()



"""
Apache-2.0 license
https://github.com/hafriedlander/stable-diffusion-grpcserver/blob/main/sdgrpcserver/services/generate.py
https://github.com/parlance-zz/g-diffuser-bot/tree/g-diffuser-bot-beta2
_handleImageAdjustment
"""

functbl = {
    "gaussian": gaussian_noise,
    "edge_pad": edge_pad,
    "cv2_ns": cv2_ns,
    "cv2_telea": cv2_telea,
}


def prepare_scheduler(scheduler):
    if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
        new_config = dict(scheduler.config)
        new_config["steps_offset"] = 1
        scheduler._internal_dict = FrozenDict(new_config)
    return scheduler


def load_example_yaml(example_name, yaml_path):
    with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
    yaml_data = None
    for d in data:
        if d['name'] == example_name:
            yaml_data = d
            break
    return yaml_data


def merge_frames(all_rundir, fps=10, save_dir=None, is_forward=False, save_depth=False, save_gif=True, DYNAMIC_MODE = None):
    """
    Merge frames from multiple run directories into a single directory with continuous naming.
    
    Parameters:
        all_rundir (list of pathlib.Path): Directories containing the run data.
        save_dir (pathlib.Path): Directory where all frames should be saved.
    """

    # Ensure save_dir/frames exists
    save_frames_dir = save_dir / 'frames'
    save_frames_dir.mkdir(parents=True, exist_ok=True)

    if save_depth:
        save_depth_dir = save_dir / 'depth'
        save_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize a counter for the new filenames
    global_counter = 0
    
    # Iterate through all provided run directories
    if is_forward:
        all_rundir = all_rundir[::-1]
    for rundir in all_rundir:
        # Ensure the rundir and the frames subdir exist
        if not rundir.exists():
            print(f"Warning: {rundir} does not exist. Skipping...")
            continue
        
        frames_dir = rundir / 'images' / 'frames'
        if not frames_dir.exists():
            print(f"Warning: {frames_dir} does not exist. Skipping...")
            continue

        if save_depth:
            depth_dir = rundir / 'images' / 'depth'
            if not depth_dir.exists():
                print(f"Warning: {depth_dir} does not exist. Skipping...")
                continue
        
        # Get all .png files in the frames directory, assuming no nested dirs
        frame_files = sorted(frames_dir.glob('*.png'), key=lambda x: int(x.stem))
        if save_depth:
            depth_files = sorted(depth_dir.glob('*.png'), key=lambda x: int(x.stem))
        
        # Copy and rename each file
        for i, frame_file in enumerate(frame_files):
            # Form the new path and copy the file
            new_frame_path = save_frames_dir / f"{global_counter}.png"
            shutil.copy(str(frame_file), str(new_frame_path))

            if save_depth:
                # Form the new path and copy the file
                new_depth_path = save_depth_dir / f"{global_counter}.png"
                shutil.copy(str(depth_files[i]), str(new_depth_path))
            
            # Increment the global counter
            global_counter += 1
    
    last_keyframe_name = 'kf1.png' if is_forward else 'kf2.png'
    last_keyframe = all_rundir[-1] / 'images' / last_keyframe_name
    new_frame_path = save_frames_dir / f"{global_counter}.png"
    shutil.copy(str(last_keyframe), str(new_frame_path))

    if save_depth:
        last_depth_name = 'kf1_depth.png' if is_forward else 'kf2_depth.png'
        last_depth = all_rundir[-1] / 'images' / last_depth_name
        new_depth_path = save_depth_dir / f"{global_counter}.png"
        shutil.copy(str(last_depth), str(new_depth_path))

    frames = []
    for frame_file in sorted(save_frames_dir.glob('*.png'), key=lambda x: int(x.stem)):
        frame_image = Image.open(frame_file)
        frame = ToTensor()(frame_image).unsqueeze(0)
        frames.append(frame)

    if save_depth:
        depth = []
        for depth_file in sorted(save_depth_dir.glob('*.png'), key=lambda x: int(x.stem)):
            depth_image = Image.open(depth_file)
            depth_frame = ToTensor()(depth_image).unsqueeze(0)
            depth.append(depth_frame)

    video = (255 * torch.cat(frames, dim=0)).to(torch.uint8).detach().cpu()
    video_reverse = (255 * torch.cat(frames[::-1], dim=0)).to(torch.uint8).detach().cpu()

    save_video(video, save_dir / "output.mp4", fps=fps, save_gif=save_gif)
    save_video(video_reverse, save_dir / "output_reverse.mp4", fps=fps, save_gif=save_gif)

    if save_depth:
        depth_video = (255 * torch.cat(depth, dim=0)).to(torch.uint8).detach().cpu()
        depth_video_reverse = (255 * torch.cat(depth[::-1], dim=0)).to(torch.uint8).detach().cpu()

        save_video(depth_video, save_dir / "output_depth.mp4", fps=fps, save_gif=save_gif)
        save_video(depth_video_reverse, save_dir / "output_depth_reverse.mp4", fps=fps, save_gif=save_gif)

    if DYNAMIC_MODE is not None:
        if DYNAMIC_MODE == "KeyFrameDynamic_1024_DynCraft":
            id_kf_dynamic = 0
            combined_frames = [] 
            for rundir in all_rundir:
                space_frames_dir = rundir / 'images' / 'frames'
                for space_frame_file in sorted(space_frames_dir.glob('*.png'), key=lambda x: int(x.stem), reverse=True):
                    space_frame_image = Image.open(space_frame_file)
                    space_frame = ToTensor()(space_frame_image).unsqueeze(0)
                    t, c, h, w = space_frame.shape 
                    # 创建一个宽度为 10 像素的蓝色条带  
                    # 蓝色为 (0, 0, 255)
                    blue_strip = torch.ones((t, c, h, 10))  
                    blue_strip[:, 2, :, :] = 255  
                    combined_frames.append(torch.cat((space_frame, blue_strip, space_frame), dim=3))
                last_space_frame = space_frame
                time_frames_dir = rundir / '../' / (str(id_kf_dynamic) + '_kf_dynamic') / 'result' / 'samples_separate' / 'frames'
                for time_frame_file in sorted(time_frames_dir.glob('*.png'), key=lambda x: int(x.stem), reverse=True):
                    time_frame_image = Image.open(time_frame_file)
                    time_frame = ToTensor()(time_frame_image).unsqueeze(0)
                    t, c, h, w = time_frame.shape 
                    # 创建一个宽度为 10 像素的红色条带  
                    # 红色为 (255, 0, 0)
                    red_strip = torch.ones((t, c, h, 10))  
                    red_strip[:, 0, :, :] = 255  
                    combined_frames.append(torch.cat((last_space_frame, red_strip, time_frame), dim=3))
                id_kf_dynamic += 1
                video = (255 * torch.cat(combined_frames, dim=0)).to(torch.uint8).detach().cpu()
                save_video(video, save_dir / ("campare_output_"+ DYNAMIC_MODE + ".mp4"), fps=10, save_gif=False)

def merge_keyframes(all_keyframes, save_dir, save_folder='keyframes', fps=1):
    """
    Save a list of PIL images sequentially into a directory.

    Parameters:
        all_keyframes (list): A list of PIL Image objects.
        save_dir (Path): A pathlib Path object indicating where to save the images.
    """
    # Ensure that the save_dir exists
    save_path = save_dir / save_folder
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save each keyframe with a sequential filename
    for i, frame in enumerate(all_keyframes):
        frame.save(save_path / f'{i}.png')

    all_keyframes = [ToTensor()(frame).unsqueeze(0) for frame in all_keyframes]
    all_keyframes = torch.cat(all_keyframes, dim=0)
    video = (255 * all_keyframes).to(torch.uint8).detach().cpu()
    video_reverse = (255 * all_keyframes.flip(0)).to(torch.uint8).detach().cpu()

    save_video(video, save_dir / "keyframes.mp4", fps=fps)
    save_video(video_reverse, save_dir / "keyframes_reverse.mp4", fps=fps)

def KeyFrameDynamic(keyframe, save_dir, custom_mode, dynamic_mode, inpainting_entities):
    """
    create a short dynamic video for the given keyframe, where the keyframe is the starting frame.

    Parameters:
        keyframe (image): the PIL Image object of the keyframe.
        save_dir (Path): A pathlib Path object indicating where to save the images.
    
    Output:
        lastframe (image): the PIL Image object of the last frame of the video: 
    """
    # Ensure that the save_dir exists
    prompt_path = save_dir/'img_prompt'
    prompt_path.mkdir(parents=True, exist_ok=True)
    result_path = save_dir/'result'
    result_path.mkdir(parents=True, exist_ok=True)
    # Save keyframe and a dummy prompt.txt file in folder "img_prompt"
    keyframe.save(prompt_path / 'keyframe.png')
    dummy_prompt_path = prompt_path / 'test_prompts.txt'  
    with dummy_prompt_path.open('w') as file:  
        file.write('This is a dummy prompt file.')  

    script_path = script_path_dict[dynamic_mode]
    # 使用 subprocess.run 来执行脚本 
    if dynamic_mode == "KeyFrameDynamic_1024_DynCraft":
        result = subprocess.run(["bash", script_path, str(prompt_path), str(result_path), str(custom_mode)], capture_output=True, text=True)  
    elif dynamic_mode == "KeyFrameDynamic_1024_svd":
        result = subprocess.run(["bash", script_path, str(prompt_path/"keyframe.png"), str(result_path), str(custom_mode)], capture_output=True, text=True)  
    elif dynamic_mode == "KeyFrameDynamic_EasyAnimate":
        result = subprocess.run(["bash", script_path, str(prompt_path/"keyframe.png"), str(result_path), inpainting_entities], capture_output=True, text=True)  
    else:
        raise NotImplementedError(f"Dynamic mode {dynamic_mode} is not implemented.")
    # 打印脚本的输出  
    print("标准输出:", result.stdout)  
    print("标准错误:", result.stderr)  
    
    # 检查脚本的返回码  
    if result.returncode == 0:  
        print("脚本执行成功")  
    else:  
        print("脚本执行失败，返回码:", result.returncode)  

    # load the last frame 
    if dynamic_mode == "KeyFrameDynamic_1024_DynCraft":
        lastframe = Image.open(result_path / "samples_separate/frames/15.png").convert('RGB') 
    elif dynamic_mode == "KeyFrameDynamic_1024_svd":
        lastframe = Image.open(result_path / "samples_separate/24.png").convert('RGB') 
    elif dynamic_mode == "KeyFrameDynamic_EasyAnimate":
        lastframe = Image.open(result_path / "47.png").convert('RGB')  # TODO change it to adaptive frame length
    else:
        raise NotImplementedError(f"Dynamic mode {dynamic_mode} is not implemented.")   

    if False in [True, "run noGPT4Dyn for compare"]:
        result_noGPT4Dyn_path = save_dir/'result_noGPT4Dyn'
        result_noGPT4Dyn_path.mkdir(parents=True, exist_ok=True)
        script_path = script_path_dict["KeyFrameDynamic_EasyAnimate_noGPT4Dyn"]
        result = subprocess.run(["bash", script_path, str(prompt_path/"keyframe.png"), str(result_noGPT4Dyn_path)], capture_output=True, text=True)  
        # 打印脚本的输出  
        print("标准输出:", result.stdout)  
        print("标准错误:", result.stderr)  

        # 检查脚本的返回码  
        if result.returncode == 0:  
            print("脚本执行成功")  
        else:  
            print("脚本执行失败，返回码:", result.returncode)   
    
    return lastframe


def svd_inpainting(save_dir, custom_mode_from_wonderJourney):
    """
    inpaint the warped keyframe interpolation seqence with svd model

    Parameters:
        save_dir (Path): A pathlib Path object indicating where to save the images.
    
    Output:
        lastframe (image): the PIL Image object of the last frame of the video: 
    """

    # set that the paths 
    origin_frame_dir = save_dir/ "images" / "frames_for_svd_inpainting"
    origin_frame_dir.mkdir(exist_ok=True, parents=True)
    origin_mask_dir = save_dir/ "images" / "masks"
    origin_mask_dir.mkdir(exist_ok=True, parents=True)
    input_save_dir = save_dir/  "images" / "warp"
    input_save_dir.mkdir(exist_ok=True, parents=True)
    output_frame_dir = save_dir/ "images" / "frames_after_svd_inpainting"
    output_frame_dir.mkdir(exist_ok=True, parents=True)

    # set script path
    script_path = script_path_dict["svd_inpainting"]


    # 使用 subprocess.run 来执行脚本  
    result = subprocess.run(["bash", script_path, str(custom_mode_from_wonderJourney), str(origin_frame_dir), str(origin_mask_dir), str(input_save_dir), str(output_frame_dir)], capture_output=True, text=True)  
    
    # 打印脚本的输出  
    print("标准输出:", result.stdout)  
    print("标准错误:", result.stderr)  
    
    # 检查脚本的返回码  
    if result.returncode == 0:  
        print("脚本执行成功")  
    else:  
        print("脚本执行失败，返回码:", result.returncode)  
        
    # load the last frame 
    lastframe = Image.open(output_frame_dir / "000024.png").convert('RGB')

    return lastframe
