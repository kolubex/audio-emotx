from pathlib import Path
import torch
import os
import utils.mg_utils as utils

class clip_audio_backbone(object):
    """
    Class to convert videos to audios and send audio tensors.
    """
    def __init__(self, config, movie_ids):
        """
        Args:
            config (dict): Configuration file.
            movie_ids (list): List of movie ids.
        """
        self.CLIP_AUDIO_PATH = Path(config['clip_audio_path']) / Path(config["audio_feat_type"])
        self.movie_ids = movie_ids
        self.audios = dict()
        self.time_audio = dict()
        self.device = torch.device("cuda:{}".format(config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        self.read_audios()
            

    def read_audios(self):
        """
        Read the videos, convert them to audios using ffmpeg and store them in a dictionary.
        """
        for movie_id in self.movie_ids:
            total_files = os.listdir(self.CLIP_AUDIO_PATH/movie_id)
            audio_files = [file for file in total_files if file.endswith("chunk1.wav")]
            for audio_file in audio_files:
                audio_path = self.CLIP_AUDIO_PATH/movie_id/audio_file
                scene_name = audio_file.split("_")[0]
                key = Path(movie_id)/Path(scene_name)
                self.audios[key] = audio_path

    
    def get_audio(self, movie_scene_id):
        """
        Args:
            movie_id (str): Movie id.
            scene_name (str): Scene name.
        """
        # print(self.audios.keys())
        return [self.audios[movie_scene_id]]

    