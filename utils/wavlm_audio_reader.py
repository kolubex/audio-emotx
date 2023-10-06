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
        self.CLIP_AUDIO_PATH = Path(config['clip_audio_path'])
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
            # filter out files that end with .mp4
            video_files = [file for file in total_files if file.endswith(".mp4")]
            for video_file in video_files:
                # if .wav of video file is not present, convert it to .wav
                audio_file = video_file[:-4]+".wav"
                video_path = self.CLIP_AUDIO_PATH/movie_id/video_file
                audio_path = self.CLIP_AUDIO_PATH/movie_id/audio_file
                if audio_file not in total_files:
                    # convert it such that has 1 channel and 16kHz sampling rate
                    os.system("ffmpeg -i {} -ac 1 -ar 16000 {}".format(video_path, audio_path))
                scene_name = ".".join(audio_file.split('.')[:-1]) # as scene_names have '.'
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

    