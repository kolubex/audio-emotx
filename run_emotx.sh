#!/bin/bash
#SBATCH -A kolubex
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH -w gnode056
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home2/kolubex/audio_emotx/audio-emotx/logs/run_emotx_audio.log
#SBATCH --mail-user lakshmipathi.balaji@research.iiit.ac.in
#SBATCH --mail-type ALL

echo "Starting the training process"
cd /home2/kolubex/audio_emotx/audio-emotx/
# /home2/kolubex/.miniconda3/envs/emotx/bin/python3 trainer.py wandb.logging=True num_workers=8
python3 trainer.py wandb.logging=True num_cpus=6 seed=1