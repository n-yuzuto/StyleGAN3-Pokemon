#!/bin/bash

#$-l rt_AF=1
#$-l h_rt=36:00:00
#$-j y
#$-cwd
# ジョブ名
#$-N job_name
#$-m a
#$-m b
#$-m e
#$-v GPU_COMPUTE_MODE=1


# 標準出力先
#$-o logs/stdout_1.txt

# qrsh -g gcc50441 -l rt_AF=1  -l h_rt=24:00:00

source /etc/profile.d/modules.sh
module load cuda/11.0/11.0.3
module load cudnn/8.0/8.0.5
module load gcc/11.2.0
module load python/3.8/3.8.13

source ~/venv_0426/bin/activate
echo "python version is"
python3 -V

# git clone https://github.com/n-yuzuto/stylegan3-pokemon.git
cd stylegan3-pokemon

pip3 install click
# pip3 install -r requirements.txt
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

echo "working directory is {$PWD}"
# python3 dataset_tool.py --source=/home/acd14209pi/stylegan3-pokemon/pokemon_data --dest=/home/acd14209pi/stylegan3-pokemon/datasets/pokemon-256x256.zip --resolution=256x256

python3 check_cuda_version.py
python3 train.py --outdir=/home/acd14209pi/stylegan3-pokemon/training-runs --cfg=stylegan3-t --data=/home/acd14209pi/stylegan3-pokemon/datasets/pokemon-256x256.zip --cfg=stylegan3-t --gpus=8 --batch=32 --gamma=2