from PIL import Image
import torch
from tqdm import tqdm
from torchvision.transforms import ToTensor
from pathlib import Path
from torchvision.io import write_video
import imageio


def save_video(video, path, fps=10, save_gif=True):
    video = video.permute(0, 2, 3, 1)
    video_codec = "libx264"
    video_options = {
        "crf": "30",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",
    }
    write_video(str(path), video, fps=fps, video_codec=video_codec, options=video_options)
    if not save_gif:
        return
    video_np = video.cpu().numpy()
    gif_path = str(path).replace('.mp4', '.gif')
    imageio.mimsave(gif_path, video_np, duration=1000/fps, loop=0)

## general config
REVERSE = False
HAS_SPACE_DIR__FRAMES_AFTER_SVD_INPAINTING = True
HAS_SPACE_DIR__FRAMES_FOR_SVD_INPAINTING = False
HAS_SPACE_DIR__FRAMES_ORIGIN_WONDERJOURNEY = False

## dynamic config
DYNAMIC_MODE = "KeyFrameDynamic_EasyAnimate"

# interpolation model
INTERPOLATION_MODEL = "EasyAnimate"

output_dir = "output"

save_dir = Path("./"+output_dir)
import os
case_directories = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

for case_name in case_directories:

    source_dir = Path("./"+output_dir+"/"+case_name)

    all_rundir_raw = [str(inter_dir) for inter_dir in sorted(source_dir.glob('Interp*'), reverse=False) ]
    all_rundir = []
    for path in all_rundir_raw:
        all_rundir.append(Path(path))


    ## iterate all dirs to combine frames
    id_kf_dynamic = 0
    combined_frames = [] 
    print(str(all_rundir_raw))
    # for dir_idx in tqdm(range(0, 10)):
    for dir_idx in tqdm(range(0, len(all_rundir) )):
        if len(all_rundir) != 10:
            print("length of all_rundir is not 10: " + str(all_rundir))
            # raise NotImplementedError
        rundir = all_rundir[dir_idx]
        print(str(rundir))

        show_list = []

        if HAS_SPACE_DIR__FRAMES_AFTER_SVD_INPAINTING:
            space_frames_after_svd_inpainting = rundir / 'images' / 'frames_after_svd_inpainting'
            if INTERPOLATION_MODEL == "SVD":
                list_space_frames_after_svd_inpainting = sorted(space_frames_after_svd_inpainting.glob('*.png'), key=lambda x: int(x.stem), reverse=True)
            else:
                list_space_frames_after_svd_inpainting = sorted(space_frames_after_svd_inpainting.glob('*.png'), key=lambda x: int(x.stem), reverse=False)
            show_list.append(list_space_frames_after_svd_inpainting)

        if HAS_SPACE_DIR__FRAMES_FOR_SVD_INPAINTING:
            space_frames_for_svd_inpainting = rundir / 'images' / 'frames_for_svd_inpainting'
            list_space_frames_for_svd_inpainting = sorted(space_frames_for_svd_inpainting.glob('*.png'), key=lambda x: int(x.stem), reverse=True)
            show_list.append(list_space_frames_for_svd_inpainting)

        if HAS_SPACE_DIR__FRAMES_ORIGIN_WONDERJOURNEY:
            space_frames_origin_wonderjourney = rundir / 'images' / 'frames'
            list_space_frames_origin_wonderjourney = sorted(space_frames_origin_wonderjourney.glob('*.png'), key=lambda x: int(x.stem), reverse=True)
            # make sure origin_wonderjourney has the same number of frames as other two mode (by copy the first frame of other modes)
            if HAS_SPACE_DIR__FRAMES_FOR_SVD_INPAINTING:
                list_space_frames_origin_wonderjourney = [list_space_frames_for_svd_inpainting[0]] + list_space_frames_origin_wonderjourney
            elif HAS_SPACE_DIR__FRAMES_AFTER_SVD_INPAINTING:
                list_space_frames_origin_wonderjourney = [list_space_frames_after_svd_inpainting[0]] + list_space_frames_origin_wonderjourney
            show_list.append(list_space_frames_origin_wonderjourney)


        for frame_idx in range(0, len(show_list[0])):
            concat_list = []
            for show_idx in range(0, len(show_list)):
                image = Image.open(show_list[show_idx][frame_idx])
                frame = ToTensor()(image).unsqueeze(0)
                t, c, h, w = frame.shape 
                concat_list.append(frame)

                if show_idx < len(show_list) - 1:
                    # 创建一个宽度为 10 像素的蓝色条带  
                    # 蓝色为 (0, 0, 255)
                    blue_strip = torch.ones((t, c, h, 10))  
                    blue_strip[:, 2, :, :] = 255    
                    concat_list.append(blue_strip)

            combined_frames.append(torch.cat(concat_list, dim=3))

            # if frame_idx == len(show_list[0]) -1:
            #     last_space_frame = show_list[show_idx][frame_idx]

        if DYNAMIC_MODE == "no_dynamic":
            continue
        elif DYNAMIC_MODE == "KeyFrameDynamic_1024_DynCraft":
            time_frames_dir = rundir / '../' / (str(id_kf_dynamic) + '_kf_dynamic') / 'result' / 'samples_separate' / 'frames'
        elif DYNAMIC_MODE == "KeyFrameDynamic_1024_svd":
            time_frames_dir = rundir / '../' / (str(id_kf_dynamic) + '_kf_dynamic') / 'result' / 'samples_separate' 
        elif DYNAMIC_MODE == "KeyFrameDynamic_EasyAnimate": 
            time_frames_dir = rundir / '../' / (str(id_kf_dynamic) + '_kf_dynamic') / 'result' 

        show_list = []

        if HAS_SPACE_DIR__FRAMES_AFTER_SVD_INPAINTING:
            list_space_frames_after_svd_inpainting =  sorted(time_frames_dir.glob('*.png'), key=lambda x: int(x.stem), reverse=False)
            show_list.append(list_space_frames_after_svd_inpainting)

        if HAS_SPACE_DIR__FRAMES_FOR_SVD_INPAINTING:
            list_space_frames_for_svd_inpainting =  sorted(time_frames_dir.glob('*.png'), key=lambda x: int(x.stem), reverse=False)
            show_list.append(list_space_frames_for_svd_inpainting)

        if HAS_SPACE_DIR__FRAMES_ORIGIN_WONDERJOURNEY:
            list_space_frames_origin_wonderjourney =  sorted(time_frames_dir.glob('*.png'), key=lambda x: int(x.stem), reverse=False)
            show_list.append(list_space_frames_origin_wonderjourney)

        for frame_idx in range(0, len(show_list[0])):
            concat_list = []
            for show_idx in range(0, len(show_list)):
                image = Image.open(show_list[show_idx][frame_idx])
                frame = ToTensor()(image).unsqueeze(0)
                t, c, h, w = frame.shape 
                concat_list.append(frame)

                if show_idx < len(show_list) - 1:
                    # 创建一个宽度为 10 像素的红色条带  
                    # 红色为 (255, 0, 0)
                    red_strip = torch.ones((t, c, h, 10))  
                    red_strip[:, 0, :, :] = 255  
                    concat_list.append(red_strip)

            combined_frames.append(torch.cat(concat_list, dim=3))

        # for time_frame_file in sorted(time_frames_dir.glob('*.png'), key=lambda x: int(x.stem), reverse=False):
        #     time_frame_image = Image.open(time_frame_file)
        #     time_frame = ToTensor()(time_frame_image).unsqueeze(0)
        #     t, c, h, w = time_frame.shape 
        #     # # 创建一个宽度为 10 像素的红色条带  
        #     # # 红色为 (255, 0, 0)
        #     # red_strip = torch.ones((t, c, h, 10))  
        #     # red_strip[:, 0, :, :] = 255  
            # combined_frames.append(torch.cat((last_space_frame, red_strip, time_frame), dim=3))
        id_kf_dynamic += 1

    if REVERSE:
        combined_frames.reverse()  
    video = (255 * torch.cat(combined_frames, dim=0)).to(torch.uint8).detach().cpu()
    # Label_as_final = "final_"
    save_video(video, save_dir / (case_name+ ".mp4"), fps=24, save_gif=False)