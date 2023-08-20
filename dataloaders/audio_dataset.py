from pathlib import Path
import numpy
import torch

class audio_dataset(object):
    """
    Dataset class is to load audio features for given movie ids.

    TODO:
        - make a class in utils to read the audio features from the given path.
            - audio_feat_reader => you dont need it its only for srt with word based features.
        - update the config_file with these things.
            - audio_feat_dim
            - audio_feat_model(avgpool and finetuned)
            - max_features_audio
            - audio_feat_dir, one for avgpool and one for finetuned(in future)
            - audio_feat_dim(768)
        - though padding is not required for audio features, make a code to pad things for future.
    """
    def __init__(self, config, movie_ids, top_k):
        """
        Args:
            config (dict): config file
            movie_ids (list): list of movie ids
            top_k (int): top k clips to use
        """
        self.config = config
        self.feat_model = config["audio_feat_model"]
        self.data_path = Path(config["data_path"])
        self.audio_feat_path = self.data_path/config["audio_feat_dir"]
        self.max_features = config["max_audio_feats"] # as we have max_audio_feats(300) bins for each scene
        self.audio_feat_dim = config["feat_info"][self.feat_model]["audio_feat_dim"]
        self.top_k = top_k
        self.audio_feats_dir = "pretrained" if config["audio_feat_pretrained"] else "finetuned_t{}".format(self.top_k)
        # self.clip_srt_obj = audio_feat_reader(config, movie_ids)

    def get_audio_feats(self, scenes):
        """
        Description:
            It doesn't need padding if 300 dim audio features are already extracted for each scene.
            It would need padding if we dont have audio features of 300 tokens for each scene.
        Args:
            scenes (list): list of scenes(with movie ids)
        Returns:
            audio_feats (torch.Tensor): audio features for given scenes. Shape (max_features, audio_feat_dim)
            times (torch.Tensor): timestamp for individual individual audio features. Shape (max_features,)
            audio_pad_mask (torch.Tensor): padding mask for audio features. Shape (max_features,)
        """
        # audio_pad_mask = torch.zeros(self.max_features,)
        # audio_feats = torch.empty((0, self.audio_feat_dim))
        # times = torch.empty((0,))
        # for scene in scenes:
        #     # print("scene: ", scene)
        #     # tt0212338/scene-036.ss-0343.es-0348
        #     scene_path = str(str(scene).split("/")[1])
        #     with open(self.audio_feat_path/self.audio_feats_dir/scene/(str(scene_path)+".npy"), 'rb') as f:
        #         feats = numpy.load(f)
        #         # squeeze the 1st dimension of numpy array
        #         feats = numpy.squeeze(feats, axis=0)
        #         # time is the value of 2nd dimension of the numpy array
        #         # give similar shape as srt but here time means a tensor of number of audio features (3rd dimension of numpy array) 
        #         # multiplied by 1/3.
        #         time = torch.zeros(feats.shape[0])
        #         for i in range(feats.shape[0]):
        #             time[i] = (i*1/3) - 0.01 # to make sure that it falls into appropriate bin
        #     if len(feats):
        #         audio_feats = torch.cat([audio_feats, torch.tensor(feats)], dim=0)
        #         times = torch.cat([times, torch.tensor(time)], dim=0)
        # audio_feats = audio_feats[:self.max_features, :]
        # times = times[:self.max_features]
        # audio_pad_mask[audio_feats.shape[0]:] = 1
        # pad_len = self.max_features - audio_feats.shape[0]
        # audio_feats = torch.cat([audio_feats, torch.zeros(pad_len, audio_feats.shape[-1])])
        # times = torch.cat([times, torch.zeros(pad_len)])
        # return audio_feats, times, audio_pad_mask
        audio_pad_mask = torch.zeros(self.max_features,)
        audio_feats = torch.empty((self.max_features, self.audio_feat_dim))
        times = torch.empty((self.max_features,))
        for scene in scenes:
            # print("scene: ", scene)
            # tt0212338/scene-036.ss-0343.es-0348
            scene_path = str(str(scene).split("/")[1])
            with open(self.audio_feat_path/self.audio_feats_dir/scene/(str(scene_path)+".npy"), 'rb') as f:
                feats = numpy.load(f) # shape: (1)
                # squeeze the 1st dimension of numpy array
                feats = numpy.squeeze(feats, axis=0) # feats shape: (768, t)
                # transpose the numpy array
                feats = numpy.transpose(feats, (1, 0)) # feats shape: (t, 768)
                # time is the value of 2nd dimension of the numpy array
                # give similar shape as srt but here time means a tensor of number of audio features (3rd dimension of numpy array) 
                # multiplied by 1/3.
                time = torch.zeros(feats.shape[0])
                for i in range(feats.shape[0]):
                    time[i] = (i*1/3) - 0.01
            if len(feats):
                # print("feats.shape: ", feats.shape)
                # print("audio_feats.shape: ", audio_feats.shape)
                audio_feats = torch.cat([audio_feats, torch.tensor(feats)], dim=0)
                times = torch.cat([times, torch.tensor(time)], dim=0)
        audio_feats = audio_feats[:self.max_features, :]
        times = times[:self.max_features]
        audio_pad_mask[audio_feats.shape[0]:] = 1
        return audio_feats, times, audio_pad_mask
