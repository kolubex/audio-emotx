from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from utils.wavlm_audio_feat_reader import clip_audio
import numpy as np
import torch
import yaml
import utils.mg_utils as utils


class dataset_audio(Dataset):
    """
    Dataset class to load subtitles primarily as an input to RoBERTa model.
    """
    def __init__(self, config, movie_ids, split_type, emo2id=dict()):
        """
        Args:
            config (dict): config dictionary
            movie_ids (list): list of movie ids
            split_type (str): Split type. Can be train, val or test.
            emo2id (dict): dictionary mapping emotions to ids
        """
        self.CLIP_AUDIO_FEAT_PATH = Path(config['clip_wavlm_feats_path'])
        self.CLIP_AUDIO_FEAT_PATH = self.CLIP_AUDIO_FEAT_PATH / Path(config["audio_feat_type"]+"_npy_files")
        self.movie_ids = movie_ids
        self.split_type = split_type
        self.clip_audios = clip_audio(config, movie_ids)
        self.movie_graphs = utils.read_mg_pkl(config['pkl_path'], config['pkl_name'])
        self.custom_mapping = None if not config["use_emotic_mapping"] else utils.get_emotic_emotions_map(config["emotic_mapping_path"])
        self.top_k_emotions = self.get_top_k_emotions(config["top_k"], config["use_emotic_mapping"])
        self.top_k = len(self.top_k_emotions) if not config["use_emotic_mapping"] else len(self.custom_mapping)
        # self.tokenizer = self.load_tokenizer(config["hugging_face_cache_path"]) # todo: define a tokenizer for duplicating every 50th feature.
        # ! the above todo is handled in clip_audio.process_audio()
        # ? do other thing like padding in there.
        self.emo2id = emo2id
        self.cgs = dict()
        self.scenes = list()
        self.targets = list()
        self.device = torch.device("cuda:{}".format(config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        self.prepare_dataset(config["use_emotic_mapping"])

    def get_top_k_emotions(self, top_k, emotic_mapped=False):
        """
        Get top k emotions from the movie graphs.
        Operated in two modes: emotic_mapped and non-emotic_mapped.
        In emotic_mapped mode, the MovieGraphs emotions are mapped to the emotic emotions.
        In non-emotic_mapped mode, the MovieGraphs emotions are used as is, based on top-k values.

        Args:
            top_k (int): top-k emotions to be considered
            emotic_mapped (bool): boolean flag to indicate whether to map the MovieGraphs emotions to emotic emotions

        Returns:
            top_k_emotions (list): list of top k emotions
        """
        top_k_emotions = list()
        if emotic_mapped:
            for _, emos in self.custom_mapping.items():
                top_k_emotions += emos
        else:
            top_k_emotions = utils.get_top_k_emotions(self.movie_graphs, top_k)
        return top_k_emotions


    def add_padding(self, audio_feats):
        """
        Add padding to the audio_feats got from wavlm to match 5100 along temporal dimension.

        Args:
            audio_feats (list): list of tensors of shape (1, audio_feat_dim, scene_length*(50)-1)
            # ! for memory management you only get paths to the numpy files.

        Returns:
            dict:
                audio_feats (list): list of tensors of shape (1, audio_feat_dim, 5100)
                masks (list): list of tensors of shape (1, 5100)
        """
        duplicated_audio_feats = []
        for audio_feat in audio_feats:
            audio_feat = np.load(audio_feat[0])
            audio_feat = torch.from_numpy(audio_feat)
            audio_feat = audio_feat.squeeze(0).transpose(0,1)
            for i in range(0, audio_feat.shape[0], 50):
                dup = audio_feat[i,:].unsqueeze(0)
                audio_feat = torch.cat((audio_feat, dup), dim=0)
            audio_feat = audio_feat.transpose(0, 1).unsqueeze(0)
            duplicated_audio_feats.append(audio_feat)
        audio_feats = duplicated_audio_feats
        # audio_feats shape is 1, 768, scene_length*(50)-1+duplicated
        # I want to pad it to 1, 768, 5100
        audio_len = [int(audio_feat.shape[2]/17) for audio_feat in audio_feats]
        masks = [torch.zeros((1,300)) for _ in range(len(audio_feats))] 
        for i in range(len(audio_feats)):
            if(audio_len[i] >= 300):
                pass
            else:
                masks[i][:,int(audio_len[i]):] = 1
        pad_token = torch.zeros_like(audio_feats[0][:,:,0]).unsqueeze(2)
        audio_feats = [torch.cat((audio_feat, pad_token.repeat(1, 1, 5100-audio_feat.shape[2])), dim=2) if audio_feat.shape[2] < 5100 else audio_feat[:,:,:5100] for audio_feat in audio_feats]
        # remove the first dimension in audio_feats and masks
        audio_feats = [audio_feat.squeeze(0) for audio_feat in audio_feats]
        masks = torch.stack(masks)
        audio_feats = torch.stack(audio_feats)
        return {"audio_feats": audio_feats, "masks": masks}


    def filter_cgs(self):
        """
        Filter the clip graphs from movies to retain only the clips that have the top-k emotions.
        """
        all_mapped_clip_graphs = utils.get_clip_graphs(self.movie_graphs, self.movie_ids)
        for m_id, cgs in all_mapped_clip_graphs.items():
            filtered_clip_graphs, _, _ = utils.get_emo_filtered_cgs(cgs, self.top_k_emotions)
            self.cgs[m_id] = filtered_clip_graphs

    def map_emo_with_id(self, emotic_mapped=False):
        """
        Map the emotions to ids based on emotic-mapped bool flag.
        In case of emotic-mapped, 181 MovieGraphs emotions are mapped to respective 26 Emotic classes.
        In case of non-emotic-mapped, the top-k emotions are mapped to ids (based of ranking).

        Args:
            emotic_mapped (bool): boolean flag to indicate whether to map the MovieGraphs emotions to emotic emotions
        """
        if emotic_mapped:
            for ndx, pair in enumerate(self.custom_mapping.items()):
                group_label, emos = pair
                for emo in emos:
                    self.emo2id[emo] = ndx
        else:
            for ndx, emo in enumerate(self.top_k_emotions):
                self.emo2id[emo] = ndx

    def build_target_vector(self, emotions):
        """
        Build the target vector for the clip graphs.
        The target vector is a one-hot vector of size top-k (or 26 for Emotic) with the emotions present in the clip graph set to 1.

        Args:
            emotions (list): list of emotions present in the clip graph.

        Returns:
            vector (torch.tensor): target vector for the clip graph.
        """
        vector = torch.zeros(self.top_k)
        for emo in emotions:
            if emo in self.emo2id.keys():
                vector[self.emo2id[emo]] = 1
        return vector

    def collect_data(self):
        """
        Collect the emotion and scene data from the clip graphs and build the target vectors.
        """
        for m_id in self.movie_ids:
            for cg in self.cgs[m_id]:
                scenes = [Path(m_id)/vid_name[:-4] for vid_name in cg.video['fname']]
                emotions = [pair[1] for pair in utils.get_emotions_from_cg(cg)]
                target_vec = self.build_target_vector(emotions)
                if m_id == 'tt1568346' and cg.video['fname'][0] == 'scene-079.ss-0893.es-0935.mp4':
                    scenes = [Path(m_id)/'scene-079.ss-0893.es-0935']
                self.scenes.append(scenes)
                self.targets.append(target_vec)

    def get_emo2id_mapping(self):
        """
        Get the emotion to id mapping.
        """
        return self.emo2id

    def prepare_dataset(self, emotic_mapped=False):
        """
        Prepare the dataset for the split type.
        First, the clip graphs are filtered to based on top-k/emotic-mapped emotions.
        Then, the emotions are mapped to ids. For val/test sets, train's emo2id mapping is used.
        Finally, the scene and target data is collected from the clip graphs.

        Args:
            emotic_mapped (bool): boolean flag to indicate whether to map the MovieGraphs emotions to emotic emotions.
        """
        self.filter_cgs()
        if not self.emo2id.keys():
            self.map_emo_with_id(emotic_mapped)
        self.collect_data()
        print("Finished preparing {}-split\n".format(self.split_type))

    def collate(self, batches):
        """
        Collate the batch data into a dictionary.

        Args:
            batches (list): list of batches

        Returns:
            collated_data (dict): dictionary of collated data with input_ids, attention masks and targets.
        """
        audio_feats = [pair[0] for pair in batches]
        targets = torch.stack([pair[1] for pair in batches])
        padded_batch = self.add_padding(audio_feats)
        collated_data = {
            "feats": [padded_batch['audio_feats']],
            "masks": [padded_batch['masks']],
            "targets": [targets]
        }
        return collated_data

    def __getitem__(self, ndx):
        """
        Get the data item at the given index.

        Args:
            ndx (int): index of the data item

        Returns:
            audio_feats(list) : list of tensors of shape (1, audio_feat_dim, scene_length*(50)-1+duplicated)
            target (torch.tensor): target vector for the clip graph
        """
        scenes = self.scenes[ndx]
        target = self.targets[ndx]
        audio_feats = []
        for scene in scenes:
            audio_feats.append(self.clip_audios.get_audio_feat(scene)[0])
        return (audio_feats, target)

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.targets)
