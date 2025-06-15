import os
import subprocess
from tqdm import tqdm
import time
from datetime import datetime
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Run cases and measure execution time')
    parser.add_argument('--config_dir', type=str, default='config/case_run_single',
                        help='Directory containing YAML configuration files')
    return parser


def main():
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    config_dir = args.config_dir

    # 获取目录下所有.yaml文件
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    # 打印实际执行顺序
    print(yaml_files)

    # 确保time_statistics.txt文件存在并清空内容
    with open('time_statistics.txt', 'w', encoding='utf-8') as f:
        # 写入元信息
        f.write('# Time statistics for running cases\n')
        f.write('# Generated at: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write('# Format: case_name: elapsed_time_in_seconds\n')
        f.write('# ----------------------------------------\n\n')

    # 遍历每个yaml文件，并显示进度条
    for yaml_file in tqdm(yaml_files, desc="Processing files"):

        print("current case:" + yaml_file)

        # 获取去掉后缀的文件名
        current_case_name = yaml_file.replace('.yaml', '')
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行第一个命令
        run_command = f"python run.py --example_config {config_dir}/{yaml_file}"
        subprocess.run(run_command, shell=True)
        
        # # 执行第二个命令 (never use anymore) # use again in 2024.10.31
        interpolation_command = f"python interpolation_generation.py --target_dir output/{current_case_name}"
        subprocess.run(interpolation_command, shell=True)
        
        # 计算运行时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 将时间写入文件
        with open('time_statistics.txt', 'a', encoding='utf-8') as f:
            f.write(f"{yaml_file}: {elapsed_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
