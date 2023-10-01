from models.wavlm_finetuning import featExtract_finetuned_WavLM
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
        # self.audio_type = config["audio_type"]
        self.audio_path = Path(config["clip_wavlm_feats_path"]+"/"+config["audio_feat_type"]+"_npy_files")
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
            model_name = Path("WavLM_finetuned_t{}_scene.pt".format(self.top_k))
            self.model = featExtract_finetuned_WavLM(self.top_k, Path(config["saved_model_path"])/Path(config["audio_feat_type"]/model_name))
        self.model = self.model.eval().to(self.device)

    def save_feats(self, save_path, scene, feats,masks):
        """
        Saves the timestamps for every utterance and the extracted features to a pickle file.
        """
        with open(save_path/(scene+".pkl"), 'wb') as f:
            pickle.dump(masks, f)
            pickle.dump(feats, f)

    def add_padding(self, audio_feats):
        """  
        Add padding to the audio_feats got from wavlm to match 5100 along temporal dimension.

        Args:
            audio_feats (list): list of tensors of shape (1, audio_feat_dim, scene_length*(50)-1)
            # ! for memory management you only get paths to the numpy files.

        Returns:
            dict:
                audio_feats (list): list of tensors of shape (1, audio_feat_dim, 5100)
                masks (list): list of tensors of shape (1, num_timestamps(default:300))
        """
        duplicated_audio_feats = []
        for audio_feat in audio_feats:
            # ! this is for the case mentioned in help.md about a sfx file
            if(audio_feat.shape[0] == 2):
                audio_feat = audio_feat[0]
            audio_feat = audio_feat.squeeze(0).transpose(0,1)
            for i in range(0, audio_feat.shape[0], 50):
                dup = audio_feat[i,:].unsqueeze(0)
                audio_feat = torch.cat((audio_feat, dup), dim=0)
            audio_feat = audio_feat.transpose(0, 1).unsqueeze(0)
            duplicated_audio_feats.append(audio_feat)
        audio_feats = duplicated_audio_feats
        audio_len = [int(audio_feat.shape[2]/17) for audio_feat in audio_feats]
        masks = [torch.zeros((1,300)) for _ in range(len(audio_feats))] 
        for i in range(len(audio_feats)):
            if(audio_len[i] >= 300):
                audio_len[i] = 300
            else:
                masks[i][:,int(audio_len[i]):] = 1
        pad_token = torch.zeros_like(audio_feats[0][:,:,0]).unsqueeze(2)
        audio_feats = [torch.cat((audio_feat, pad_token.repeat(1, 1, 5100-audio_feat.shape[2])), dim=2) if audio_feat.shape[2] < 5100 else audio_feat[:,:,:5100] for audio_feat in audio_feats]
        # remove the first dimension in audio_feats and masks
        audio_feats = [audio_feat.squeeze(0) for audio_feat in audio_feats]
        masks = torch.stack(masks)
        audio_feats = torch.stack(audio_feats)
        audio_feats = audio_feats.unsqueeze(0).to(self.device)
        masks = masks.unsqueeze(0).to(self.device)
        return {"audio_feats": audio_feats, "masks": masks, "audio_len": audio_len}

    @torch.no_grad()
    def extract_features(self, movies):
        """
        Extracts the utterance features for the given list of movies.

        Args:
            movies (list): List of movies for which the features are to be extracted.
        """
        pst = time.perf_counter()
        # movies = ["tt1013753"]
        for movie in movies:
            save_path = self.audio_feat_save_path/movie
            if not os.path.exists(save_path):
                save_path.mkdir(parents=True, exist_ok=True)
                st = time.perf_counter()
                print("Started extracting features for: {}".format(movie), end=" | ")
                # movie -> scenen_folder/scene_name.npy
                audios_feats = [self.audio_path/movie/scene_folder/scene for scene_folder in os.listdir(self.audio_path/movie) for scene in os.listdir(self.audio_path/movie/scene_folder)]
                for audio in audios_feats:
                    times, feats = list(), list()
                    if audio and str(audio).endswith(".npy"):
                        audio_tensor = [torch.from_numpy(np.load(audio)).to(self.device)]
                        feats_masks = self.add_padding(audio_tensor)
                        feats = self.model(feats_masks["audio_feats"],feats_masks["masks"],logits=False)[0]
                        feats = feats.detach().cpu()
                    # feat_file_name = parent of posix path => a/b.py => a
                    feat_file_name= Path(audio).parent.name
                    self.save_feats(save_path, feat_file_name, feats, feats_masks["audio_len"])
                print("Completed in {:.4f} sec.".format(time.perf_counter()-st))
            print("Finished extraction in {:.4f} sec.".format(time.perf_counter()-pst))


def get_config():
    """
    Loads the config file and overrides the hyperparameters from the command line.
    """
    base_conf = OmegaConf.load("config_finetune_audio.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)


if __name__ == "__main__":
    cnfg = get_config()
    mvs = os.listdir(cnfg["clip_wavlm_feats_path"]+"/"+cnfg["audio_feat_type"]+"_npy_files")
    obj = audio_feat_extraction(cnfg)
    obj.extract_features(mvs)
