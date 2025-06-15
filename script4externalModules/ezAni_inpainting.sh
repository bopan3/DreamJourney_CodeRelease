origin_frame_dir=$1
origin_mask_dir=$2
output_frame_dir=$3
result_analyze_dir=$4


# 使用 CONDA_HOME 环境变量，如果未设置则使用默认路径
CONDA_HOME=${CONDA_HOME:-"$HOME/miniconda3"}
CONDA_SH="$CONDA_HOME/etc/profile.d/conda.sh"
# 检查 conda.sh 是否存在
if [ ! -f "$CONDA_SH" ]; then
    echo "Error: conda.sh not found at $CONDA_SH"
    echo "Please set CONDA_HOME environment variable to your conda installation path"
    echo "Example: export CONDA_HOME=/path/to/your/conda"
    exit 1
fi
source "$CONDA_SH"
conda init bash
source ~/.bashrc


conda activate easyAnimate 
python ./video_prior_models/EasyAnimate/EzAni_inference.py \
    --origin_frame_dir $origin_frame_dir \
    --origin_mask_dir $origin_mask_dir \
    --output_frame_dir $output_frame_dir \
    --result_analyze_dir $result_analyze_dir \
    --lr 0.025