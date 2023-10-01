from pathlib import Path
import numpy
import torch
import pickle
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
        self.top_k = top_k
        self.audio_feat_path = self.data_path/Path(config["audio_feat_dir"])/Path(f"""{config["audio_feat_type"]}""")
        self.max_features = config["max_audio_feats"] # as we have max_audio_feats(300) bins for each scene
        self.audio_feat_dim = config["feat_info"][self.feat_model]["audio_feat_dim"]
        self.audio_feats_dir = "pretrained" if config["audio_feat_pretrained"] else "finetuned_t_{}".format(self.top_k)
        # self.clip_srt_obj = audio_feat_reader(config, movie_ids)

    def get_feats_masks(self, scenes):
        # load from pkl file
        with open(self.audio_feat_path/"audio_feats.pkl", 'rb') as f:
            audio_feats = pickle.load(f)
            masks = pickle.load(f)
        audio_feats = audio_feats[scenes]
        masks = masks[scenes]
        return audio_feats, masks

    
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
        audio_pad_mask = torch.zeros(self.max_features,)
        audio_feats = torch.empty((0, self.audio_feat_dim))
        times = torch.empty(0,)
        for scene in scenes:
            with open(self.audio_feat_path/self.audio_feats_dir/(str(scene)+".pkl"), 'rb') as f:
                masks = pickle.load(f)
                feats = pickle.load(f)
                feats = feats.squeeze(0).transpose(0, 1)
                time = numpy.arange(1,masks[0]) * 1/3
                time = time - 0.01

            if len(feats):
                audio_feats = torch.cat([audio_feats, feats], dim=0)
                times = torch.cat([times, torch.from_numpy(time)], dim=0)
        audio_feats = audio_feats[:self.max_features, :]
        times = times[:self.max_features]
        audio_pad_mask[times.shape[0]:] = 1
        pad_len = self.max_features - times.shape[0]
        times = torch.cat([times, torch.zeros(pad_len)])
        return audio_feats, times, audio_pad_mask
