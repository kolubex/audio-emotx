#!/bin/bash
#SBATCH -A kolubex
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH -w gnode048
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home2/kolubex/audio_emotx/feat_extract/wavlm/logs/run_final_tt1568346.log
#SBATCH --mail-user lakshmipathi.balaji@research.iiit.ac.in
#SBATCH --mail-type ALL

cd /home2/kolubex/audio_emotx/audio-emotx/
git checkout audio_integration
python trainer.py