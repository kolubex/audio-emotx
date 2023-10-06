from models.wavlm_finetuning_backbone import featExtract_finetuned_WavLM
from omegaconf import OmegaConf
from pathlib import Path
from transformers import RobertaTokenizer
from utils.wavlm_audio_feat_reader import clip_audio
from dataloaders.mg_for_wavlm import dataset_audio
import numpy as np

import os
import pickle
import torch
import time
import yaml
import subprocess
import torchaudio

class audio_feat_extraction(object):
    """
    Class for extracting features from the finetuned or pretrained RoBERTa models.
    """
    def __init__(self, config):
        """
        Initializes the class with the config file.

        Args:
            config (dict): The config file with all the hyperparameters.
        """
        self.config = config
        self.audio_path = Path(config["clip_audio_path"])
        self.audio_feat_save_path = Path(config["save_feats_path"])/Path(config["audio_feat_dir"])/Path(config["audio_feat_type"])
        # if self.audio_feat_save_path directory exists remove it and create a new one
        if self.audio_feat_save_path.exists():
            os.system("rm -rf {}".format(self.audio_feat_save_path))
        self.audio_feat_save_path.mkdir(parents=True, exist_ok=True)
        self.top_k = config["top_k"] if not config["use_emotic_mapping"] else 26
        # self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=config["hugging_face_cache_path"])
        self.device = torch.device("cuda:{}".format(config["gpu_id"]) if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device("cpu")
        if not config["audio_feat_pretrained"]:
            print("Selected finetuned WavLM model for top-{} emotions".format(self.top_k))
            self.audio_feat_save_path = self.audio_feat_save_path/("finetuned_t_{}".format(self.top_k))
            model_name = Path("WavLM_finetuned_backbone_t{}_scene.pt".format(self.top_k))
            self.model = featExtract_finetuned_WavLM(Path(config["saved_model_path"])/Path(config["audio_feat_type"]/model_name))
        self.model = self.model.eval().to(self.device)

    def save_feats(self, save_path, scene, feats,masks):
        """
        Saves the timestamps for every utterance and the extracted features to a pickle file.
        """
        with open(save_path/(scene+".pkl"), 'wb') as f:
            pickle.dump(masks, f)
            pickle.dump(feats, f)

    def convert_mp4_to_wav(mp4_filepath, wav_filepath):
        """
        Convert .mp4 file to .wav format using ffmpeg through subprocess.
        """
        command = ['ffmpeg', '-i', mp4_filepath, '-ac', '1', '-ar', '16000', '-vn', wav_filepath]
        subprocess.run(command)

    @torch.no_grad()
    def extract_features(self, movies):
        """
        Extracts the utterance features for the given list of movies.

        Args:
            movies (list): List of movies for which the features are to be extracted.
        """
        pst = time.perf_counter()
        movies = ["tt1013753"]
        for movie in movies:
            save_path = self.audio_feat_save_path/movie
            if not os.path.exists(save_path):
                save_path.mkdir(parents=True, exist_ok=True)
                st = time.perf_counter()
                print("Started extracting features for: {}".format(movie), end=" | ")
                # movie -> scenen_folder/scene_name.npy
                # list all audio files that end with .wav if for a scene .mp4 file .wav doesn't exist then convert that to .wav and then extract features
                all_scene_wav_files = []
                for filename in os.listdir(self.audio_path/movie):
                    if filename.endswith(".wav") or filename.endswith(".mp4"):
                        wav_filename = filename.replace(".mp4", ".wav")
                        mp4_filepath = os.path.join(self.audio_path/movie, filename)
                        wav_filepath = os.path.join(self.audio_path/movie, wav_filename)
                        
                        # If .wav file doesn't exist, convert .mp4 to .wav
                        if not os.path.exists(wav_filepath):
                            self.convert_mp4_to_wav(mp4_filepath, wav_filepath)
                        all_scene_wav_files.append(wav_filepath)
                
                # Extract features for all the audio files in the scene
                for audio in all_scene_wav_files:
                    times, feats = list(), list()
                    try:
                        if audio and str(audio).endswith(".wav"):
                            audio_tensor = torchaudio.load(audio)[0].to(self.device)
                            # length of audio in seconds
                            feat_mask = audio_tensor.shape[1]//16000
                            # take first 15 seconds of audio
                            feat_mask = min(feat_mask, 40)
                            audio_tensor = audio_tensor[:,:feat_mask*16000]
                            print(audio_tensor.shape)
                            feats = self.model(audio_tensor.unsqueeze(0),[feat_mask],logits=False)[0]
                            feats = feats.detach().cpu()
                            feat_file_name = Path(audio).stem
                            self.save_feats(save_path, feat_file_name, feats, feat_mask)
                    except:
                        print("Error in extracting features for: {}".format(audio))
                        continue
                print("Completed in {:.4f} sec.".format(time.perf_counter()-st))
            print("Finished extraction in {:.4f} sec.".format(time.perf_counter()-pst))


def get_config():
    """
    Loads the config file and overrides the hyperparameters from the command line.
    """
    base_conf = OmegaConf.load("config_finetune_audio.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)


if __name__ == "__main__":
    cnfg = get_config()
    mvs = os.listdir(cnfg["clip_audio_path"])
    obj = audio_feat_extraction(cnfg)
    obj.extract_features(mvs)
