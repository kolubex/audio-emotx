#!/bin/bash
#SBATCH -A research
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH -w gnode049
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home2/kolubex/audio_emotx/audio-emotx/logs/slurm-%j.out
#SBATCH --mail-user lakshmipathi.balaji@research.iiit.ac.in
#SBATCH --mail-type ALL

cd /home2/kolubex/audio_emotx/audio-emotx/
/ssd_scratch/cvit/kolubex/envs/emotx/bin/python3 trainer.py batch_size=4 num_cpus=8 wandb.logging=True