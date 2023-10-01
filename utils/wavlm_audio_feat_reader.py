from pathlib import Path
import numpy as np
import os
import torch
import yaml
import utils.mg_utils as utils


class clip_audio(object):
    """
    Class to read the srt files and return the text.
    """
    def __init__(self, config, movie_ids):
        """
        Args:
            config (dict): Configuration file.
            movie_ids (list): List of movie ids.
        """
        self.CLIP_AUDIO_FEAT_PATH = Path(config['clip_wavlm_feats_path'])/Path(config["audio_feat_type"]+"_npy_files")
        self.movie_ids = movie_ids
        self.audio_feat_type = config["audio_feat_type"]
        self.audios = dict()
        self.time_audio = dict()
        # self.device = torch.device("cuda:{}".format(self.config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        self.device = "cuda:0"
        self.read_audios()

    def process_audio(self, audio_feat_file):
        """
        Duplicate every 50th feature to match out number of bins after its pooled in finetuning layers.

        Args:
            audio_feat_file (np.array): Array of shape (1, audio_feat_dim, scene_length*(50)-1) 
            # it dimension given by wavlm.

        Returns:
            (list): List of tuples containing start and end timestamp and utterance string.
        """
        audio_feat_file = torch.from_numpy(audio_feat_file).to(device=self.device)
        audio_feat_file = audio_feat_file.squeeze(0).transpose(0,1)
        for i in range(0, audio_feat_file.shape[0], 50):
            dup = audio_feat_file[i,:].unsqueeze(0)
            audio_feat_file = torch.cat((audio_feat_file, dup), dim=0)
        audio_feat_file = audio_feat_file.transpose(0, 1).unsqueeze(0)
        audio_feat_file = audio_feat_file.detach().cpu().numpy()
        # print("audio_feat_file.shape: ", audio_feat_file.shape)
        return audio_feat_file
    
    def read_audios(self):
        """
        Read the srt files and store the text in a dictionary.
        """
        for movie_id in self.movie_ids:
            audio_files = os.listdir(self.CLIP_AUDIO_FEAT_PATH/movie_id)
            for audio_file in audio_files:
                scene_name = ".".join(audio_file.split('.')[:]) # as scene_names have '.'
                audio_file_dir = audio_file
                if self.audio_feat_type == "total":
                    audio_file += ".npy"
                else:
                    audio_file = f"{self.audio_feat_type}.npy"
                key = Path(movie_id)/scene_name
                # print("key: ", key)
                # audio_feat_file = np.load(self.CLIP_AUDIO_FEAT_PATH/movie_id/audio_file_dir/audio_file)
                audio_feat_file = self.CLIP_AUDIO_FEAT_PATH/movie_id/audio_file_dir/audio_file
                # self.audios[key] = self.process_audio(audio_feat_file)\
                self.audios[key] = audio_feat_file
                # if key == "tt0109831/scene-081.ss-0488.es-0505"
                

    def get_audio_feat(self, movie_scene_id, total=True):
        """
        TODO: Plans are there to integrate feature of combinations of various parts like music, sfx...

        Args:
            movie_scene_id (Path): Path to the srt file.
            total (bool): If True, then returns the features for total audio without any seperation.

        Returns:
            (list): List of tensors of shape (1, audio_feat_dim, scene_length*(50)-1+duplicated) 
        """
        if total:
            return [self.audios[movie_scene_id]]
        else:
            return self.time_audio[movie_scene_id]
