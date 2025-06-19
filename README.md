<p align="center">
    <img src="assets/logo.png" height=150>
</p>

# DreamJourney: Perpetual View Generation with Video Diffusion Models

<div align="center">

[![a](https://img.shields.io/badge/Website-DreamJourney-blue)](https://dream-journey.vercel.app/)
[![arXiv](https://img.shields.io/badge/arXiv-xx-red)](https://arxiv.org/abs/2312.03884)
</div>




## Getting Started

### Installation
For the installation to be done correctly, please proceed only with CUDA-compatible GPU available (tested with NVIDIA A40 40GB, NVIDIA A100 80GB, NVIDIA A800 80GB). 

Clone the repo and create the environment:
```bash
git clone https://github.com/bopan3/DreamJourney_CodeRelease
cd DreamJourney_CodeRelease
mamba create --name dreamjourney_15 python=3.10
mamba activate dreamjourney_15
```

Run the following commands to install pytorch3D or follow their <a href="https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md" target="_blank">installation guide</a> (it is recommand to use mamba instead of conda for faster installation).
```bash
mamba install "pytorch-gpu==cuda118" torchvision -c conda-forge
mamba install -c iopath iopath
mamba install pytorch3d -c pytorch3d
```

Install the rest of the requirements:

```bash
mamba install transformers diffusers accelerate
mamba install matplotlib scikit-image opencv av spacy -c conda-forge
pip install openai==0.28.1
pip install timm==0.6.12
pip install pillow==9.2.0
pip install kornia
pip install einops
pip install httpx
pip install tenacity
pip install omegaconf
pip install botocore
```

Load English language model for spacy:

```bash
python -m spacy download en_core_web_sm
```

Download Midas DPT model and put it to the root directory.

```bash
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

Set up independent environment for EasyAnimate (which is used as video prior model):

```bash
## enter EasyAnimate's dir
cd video_prior_models/EasyAnimate

## download weights
mkdir models/Diffusion_Transformer
mkdir models/Motion_Module
mkdir models/Personalized_Model

wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512.tar -O models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512.tar

cd models/Diffusion_Transformer/
tar -xvf EasyAnimateV3-XL-2-InP-512x512.tar
cd ../../

## create environment for easyAnimate
conda create -n easyAnimate python=3.10
conda activate easyAnimate
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install diffusers["torch"] transformers
pip install accelerate
pip install -e ".[torch]"

## install ffmpeg
apt-get update
apt-get install ffmpeg

## go back to DreamJourney folder
cd ../../
```






Set your OpenAI api_key at LLM_CONFIG/llm_config.yaml:

```yaml
APIKEY: "your_api_key_here"
API_BASE: "your_api_base_here"
API_MODEL_NAME: "your_api_model_name_here"
```



### Run examples 

- Example config file

  To run an example, first you need to write a config. An example config `./config/village.yaml` is shown below:

  ```yaml
  runs_dir: output/56_village
  
  example_name: village
  
  seed: -1
  frames: 10
  save_fps: 10
  
  finetune_decoder_gen: True
  finetune_decoder_interp: False  # Turn on this for higher-quality rendered video
  finetune_depth_model: True
  
  num_scenes: 4
  num_keyframes: 2
  use_gpt: True
  kf2_upsample_coef: 4
  skip_interp: False
  skip_gen: False
  enable_regenerate: True
  
  debug: True
  inpainting_resolution_gen: 512
  
  rotation_range: 0.45
  rotation_path: [0, 0, 0, 1, 1, 0, 0, 0]
  camera_speed_multiplier_rotation: 0.2
  ```

  The total frames of the generated example is `num_scenes` $\times$ `num_keyframes`. You can manually adjust `rotation_path` in the config file to control the rotation state of the camera in each frame. A value of $0$ indicates moving straight, $1$ signifies a right turn, and $-1$ indicates a left turn.  

- Run

  ```bash
  python run.py --example_config config/village.yaml
  ```
  You will see results in `output/56_village/{time-string}_merged`.

### How to add more examples?

We highly encourage you to add new images and try new stuff!
You would need to do the image-caption pairing separately (e.g., using DALL-E to generate image and GPT4V to generate description).

- Add a new image in `./examples/images/`.

- Add content of this new image in `./examples/examples.yaml`.

  Here is an example:

  ```yaml
  - name: new_example
    image_filepath: examples/images/new_example.png
    style_prompt: DSLR 35mm landscape
    content_prompt: scene name, object 1, object 2, object 3
    negative_prompt: ''
    background: ''
  ```

  - **content_prompt**: "scene name", "object 1", "object 2", "object 3"

  - **negative_prompt** and **background** are optional

  For controlled journey, you need to add `control_text`. Examples are as follow:

  ```yaml
  - name: poem_jiangxue
    image_filepath: examples/images/60_poem_jiangxue.png
    style_prompt: black and white color ink painting
    content_prompt: Expansive mountainous landscape, old man in traditional attire, calm river, mountains
    negative_prompt: ""
    background: ""
    control_text: ["千山鸟飞绝", "万径人踪灭", "孤舟蓑笠翁", "独钓寒江雪"]
    
  - name: poem_snowy_evening
    image_filepath: examples/images/72_poem_snowy_evening.png
    style_prompt: Monet painting
    content_prompt: Stopping by woods on a snowy evening, woods, snow, village
    negative_prompt: ""
    background: ""
    control_text: ["Snowy Woods and Farmhouse: A secluded farmhouse, a frozen lake, a dense thicket, a quiet meadow, a chilly wind, a pale twilight, a covered bridge, a rustic fence, a snow-laden tree, and a frosty ground", "The Traveler's Horse: A restless horse, a jingling harness, a snowy mane, a curious gaze, a sturdy hoof, a foggy breath, a leather saddle, a woolen blanket, a frost-covered tail, and a patient stance", "Snowfall in the Woods: A gentle snowflake, a whispering wind, a soft flurry, a white blanket, a twinkling icicle, a bare branch, a hushed forest, a crystalline droplet, a serene atmosphere, and a quiet night", "Deep, Dark Woods in the Evening: A mysterious grove, a shadowy tree, a darkened sky, a hidden trail, a silent owl, a moonlit glade, a dense underbrush, a quiet clearing, a looming branch, and an eerie stillness"]
  ```

- Write a config `config/new_example.yaml` like `./config/village.yaml` for the new example

- Run

  ```bash
  python run.py --example_config config/new_example.yaml
  ```


