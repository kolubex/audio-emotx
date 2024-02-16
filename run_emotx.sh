#!/bin/bash
#SBATCH -A kolubex
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH -w gnode084
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home2/kolubex/audio_emotx/audio-emotx/logs/run_emotx_audio.log
#SBATCH --mail-user lakshmipathi.balaji@research.iiit.ac.in
#SBATCH --mail-type ALL

echo "Starting the training process"
cd /home2/kolubex/audio_emotx/audio-emotx/
# /home2/kolubex/.miniconda3/envs/emotx/bin/python3 trainer.py wandb.logging=True num_workers=8
# python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_music audio_feat_type=sfx gpu_id=0 & python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_music audio_feat_type=music gpu_id=1 & python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_general audio_feat_type=total gpu_id=2 & python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_general audio_feat_type=no_vocals gpu_id=3 ;
python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_general audio_feat_type=vocals gpu_id=0 & python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_general audio_feat_type=sfx gpu_id=1 & python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_general audio_feat_type=music gpu_id=2 & python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_music_and_speech audio_feat_type=total gpu_id=3 ;
python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_music_and_speech audio_feat_type=no_vocals gpu_id=0 top_k=10 & python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_music_and_speech audio_feat_type=vocals gpu_id=1 top_k=10 & python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_music_and_speech audio_feat_type=sfx gpu_id=2 top_k=10 & python3 trainer.py wandb.logging=True num_cpus=8 seed=1 audio_feat_submodel=larger_clap_music_and_speech audio_feat_type=music gpu_id=3 top_k=10 ;