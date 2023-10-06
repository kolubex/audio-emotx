from transformers import WavLMModel, AutoProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F

class finetune_WavLM_backbone(nn.Module):
    def __init__(self, config, conv_type = 'avg'):
        super(finetune_WavLM_backbone, self).__init__()
        self.kernel_size1 = config["kernel_size1"]
        self.stride1 = config["stride1"]
        self.padding1 = config["padding1"]
        self.num_seconds = config["max_audio_len"]
        self.conv_type = config["conv_type"] if conv_type != 'avg' else conv_type
        self.top_k = config["top_k"]
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", cache_dir=config["hugging_face_cache_path"])
        self.model.train()
        self.conv1d = nn.Conv1d(768, 512, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
        self.gelu = nn.GELU()
        self.linear = nn.Linear(512, self.top_k)

    def add_padding(self, audio_feats):
        """
        Add padding to the audio_feats got from wavlm to match seq_len_after_pad along temporal dimension.

        Args:
            audio_feats (list): list of tensors of shape (1, audio_feat_dim, scene_length*(50)-1)
            # ! for memory management you only get paths to the numpy files.

        Returns:
            dict:
                audio_feats (list): list of tensors of shape (1, audio_feat_dim, seq_len_after_pad)
                masks (list): list of tensors of shape (1, seq_len_after_pad)
        """
        num_bins_after_pad = self.num_seconds*3
        seq_len_after_pad = self.kernel_size1*num_bins_after_pad 
        duplicated_audio_feats = []
        for audio_feat in audio_feats:
            for i in range(0, audio_feat.shape[0], 50):
                dup = audio_feat[i,:].unsqueeze(0)
                audio_feat = torch.cat((audio_feat, dup), dim=0)
            audio_feat = audio_feat.transpose(0, 1).unsqueeze(0)
            duplicated_audio_feats.append(audio_feat)
        audio_feats = duplicated_audio_feats
        # audio_feats shape is 1, 768, scene_length*(50)-1+duplicated
        # I want to pad it to 1, 768, seq_len_after_pad
        audio_feats = [audio_feat.squeeze(0) for audio_feat in audio_feats]
        audio_feats = torch.stack(audio_feats)
        return {"audio_feats": audio_feats}
    
    def forward(self, feats, seq_len, logits=True):
        feats = feats[0]
        seq_len = seq_len[0]
        x,extract_feats = self.model(feats, return_dict=False)
        x = self.add_padding(x)
        x = self.conv1d(x["audio_feats"])
        x = self.gelu(x)
        if logits:
            denominator = (seq_len).to(x.device).unsqueeze(-1)
            denominator[denominator == 0] = 300
            x = torch.sum(x,dim=2)/denominator
            x = self.linear(x)
            activations = x
        else:
            activations = x
        return [activations]
    
class featExtract_finetuned_WavLM(nn.Module):
    def __init__(self, model_path):
        super(featExtract_finetuned_WavLM, self).__init__()
        self.model = self.get_model(model_path)

    def get_model(self, model_path):
        model = torch.load(model_path)
        model.eval()
        return model

    def forward(self, feats, masks, logits=False):
        feat = self.model(feats, masks, logits=False)
        return feat[0]