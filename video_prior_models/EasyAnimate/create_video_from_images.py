import cv2
import os
from pathlib import Path
import subprocess

def create_video_from_images(folder_path):    
    # 获取文件夹中的所有图片文件名，并按数字顺序排序    
    images = sorted([img for img in os.listdir(folder_path) if img.endswith(".png")], key=lambda x: int(x.split('.')[0]))    
    
    # 确保文件夹中有图片    
    if not images:    
        raise ValueError("No PNG images found in the specified folder.")    
    
    # 调用 FFmpeg 命令  
    image_pattern = os.path.join(folder_path, '%d.png')  
    video_path = os.path.join(folder_path, 'combined.mp4') 
    
    command = [  
        'ffmpeg',  
        '-y',  # 添加此选项以自动覆盖现有文件 
        '-framerate', '10',  
        '-i', image_pattern,  
        '-c:v', 'libx264',  
        '-pix_fmt', 'yuv420p',  
        video_path  
    ]  
    subprocess.run(command, check=True) 

# def create_video_from_images(folder_path):  
#     # 获取文件夹中的所有图片文件名，并按数字顺序排序  
#     images = sorted([img for img in os.listdir(folder_path) if img.endswith(".png")], key=lambda x: int(x.split('.')[0]))  
  
#     # 确保文件夹中有图片  
#     if not images:  
#         raise ValueError("No PNG images found in the specified folder.")  
  
#     # 读取第一张图片以获取尺寸信息  
#     first_image_path = os.path.join(folder_path, images[0])  
#     frame = cv2.imread(first_image_path)  
#     height, width, layers = frame.shape  
  
#     # 定义视频编码和创建 VideoWriter 对象  
#     video_path = os.path.join(folder_path, 'combined.avi')  # 使用 .avi 格式  
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用 MJPEG 编码  
#     video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))  
  
#     for image in images:  
#         image_path = os.path.join(folder_path, image)  
#         frame = cv2.imread(image_path)  
#         video.write(frame)  
  
#     video.release()  
#     cv2.destroyAllWindows()  

result_analyze_dir = Path("temp_folder")

create_video_from_images(str((result_analyze_dir / "0_5to50_55")))
# create_video_from_images(str((result_analyze_dir / "4to51")))


# create_video_from_images(str((result_analyze_dir / "input_frame")))
# create_video_from_images(str((result_analyze_dir / "input_mask")))
# create_video_from_images(str((result_analyze_dir / "origin_frame")))
# create_video_from_images(str((result_analyze_dir / "origin_mask")))
# create_video_from_images(str((result_analyze_dir / "output_frame")))