#!/bin/bash
#SBATCH -A kolubex
#SBATCH -n 8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -w gnode080
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home2/kolubex/audio_emotx/audio-emotx/logs/finetuned_audio_emotx_total.log
#SBATCH --mail-user lakshmipathi.balaji@research.iiit.ac.in
#SBATCH --mail-type ALL

cd /home2/kolubex/audio_emotx/audio-emotx/
python clap_feat_extractor.py