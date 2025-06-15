import os
from PIL import Image

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

if __name__ == "__main__":
    frames_dir = 'input/pad_test/images/frames_for_svd_inpainting'
    masks_dir = 'input/pad_test/images/masks_for_svd_inpainting'
    frames_padded_dir = 'input/pad_test/images/frames_for_svd_inpainting_padded'
    masks_padded_dir = 'input/pad_test/images/masks_for_svd_inpainting_padded'
    N = 4  # 你可以根据需要调整N的值

    pad_images(frames_dir, masks_dir, frames_padded_dir, masks_padded_dir, N)