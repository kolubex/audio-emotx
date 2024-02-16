#!/bin/bash
#SBATCH -A kolubex
#SBATCH -n 2
#SBATCH --nodes=1
#SBATCH --job-name=setup_gnode_audio_emotx
#SBATCH --gres=gpu:0
#SBATCH -w gnode092
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home2/kolubex/audio_emotx/audio-emotx/logs/finetuned_audio_emotx_total.log
#SBATCH --mail-user lakshmipathi.balaji@research.iiit.ac.in
#SBATCH --mail-type ALL

mkdir -p /ssd_scratch/cvit/kolubex/envs
rsync --ignore-existing -avqh ada:/share3/kolubex/envs/* /ssd_scratch/cvit/kolubex/envs 
rsync --ignore-existing -avh ada:/share3/kolubex/emotx/EmoTx_min_feats.tar.gz /ssd_scratch/cvit/kolubex/ 
# unzip 
tar -xzf /ssd_scratch/cvit/kolubex/EmoTx_min_feats.tar.gz -C /ssd_scratch/cvit/kolubex/
mkdir -p /ssd_scratch/cvit/kolubex/data/audios
rsync --ignore-existing -avh ada:/share3/haran71/total /ssd_scratch/cvit/kolubex/data/audios 
rsync --ignore-existing -avh ada:/share3/kolubex/emotx/emotx_audios/ /ssd_scratch/cvit/kolubex/data/audios 
rsync --ignore-existing -avh ada:/share3/haran71/no_vocals /ssd_scratch/cvit/kolubex/data/audios 
mkdir -p /ssd_scratch/cvit/kolubex/checkpoints
cp -r  ~/audio_emotx/mg/nx_code /ssd_scratch/cvit/kolubex/data/MovieGraph/mg/py3loader  
cp  ~/audio_emotx/mg/GraphClasses.py /ssd_scratch/cvit/kolubex/data/MovieGraph/mg/py3loader
cp  ~/audio_emotx/mg/all_movies.pkl /ssd_scratch/cvit/kolubex/data/MovieGraph/mg/py3loader
cd /home2/kolubex/audio_emotx/audio-emotx/
python audio_files_normaliser.py