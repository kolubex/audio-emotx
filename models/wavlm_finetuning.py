
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class finetune_WavLM(nn.Module):
    def __init__(self, config, ceil_mode=False, count_include_pad=True, conv_type = 'avg'):
        super(finetune_WavLM, self).__init__()
        self.kernel_size1 = config["kernel_size1"]
        self.stride1 = config["stride1"]
        self.padding1 = config["padding1"]
        self.conv_type = config["conv_type"] if conv_type != 'avg' else conv_type
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.conv1_1x1 = nn.Conv1d(768,512, kernel_size=1, stride=1, padding=0)
        self.conv2_1x1 = nn.Conv1d(512,512, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv1d(512,512, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
        self.conv3_1x1 = nn.Conv1d(512,512, kernel_size=1, stride=1, padding=0)
        self.gelu = nn.GELU()
        self.logits = nn.Linear(512, config["top_k"] )

    def forward(self, feats, masks, logits=True):
        """
        Args:
            feats (list): List of tensors with token-ids of shape (batch_size, feat_dim, seq_len) 
            * list has only one batch.
            masks (list): List of tensors of shape (batch_size, seq_len)
            logits (bool): Flag to indicate whether to return logits or not
        Returns:
            x (list): List of Tensor of shape (batch_size, num_labels) if logits=True, else (batch_size, 768)
        """
        activations = []
        feats = feats[0]; masks = masks[0]
        x = self.conv1_1x1(feats)
        x = self.gelu(x)
        x = self.conv2_1x1(x)
        x = self.gelu(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv3_1x1(x)
        x = self.gelu(x)
        if(self.conv_type == 'avg' and logits == True):
            denom = torch.argmax(masks,dim=2)
            denom[denom == 0] = 300
            x = torch.sum(x,dim=2)/denom
        if(logits == True):
            x = self.gelu(x)
            for feat in x:
                activations.append(self.logits(feat.squeeze(0).flatten()))
            activations = torch.stack(activations)
        else:
            activations = x
        return [activations]
    
class featExtract_finetuned_WavLM(nn.Module):
    def __init__(self, num_labels, model_path):
        super(featExtract_finetuned_WavLM, self).__init__()
        self.model = self.get_model(model_path)

    def get_model(self, model_path):
        model  = torch.load(model_path)
        model.eval()
        return model

    def forward(self, feats, masks,logits=False):
        return self.model(feats, masks, logits=False)
        # ! please uncomment the above line and remove things below after you retrain the 
        # ! total model with new forward function which is correct now. Coz using gpu minutes and burning trees is 
        # ! is not good for health. lol
        # activations = []
        # feats = feats[0]; masks = masks[0]
        # x = self.model.conv1_1x1(feats)
        # x = self.model.gelu(x)
        # x = self.model.conv2_1x1(x)
        # x = self.model.gelu(x)
        # x = self.model.conv1(x)
        # x = self.model.gelu(x)
        # x = self.model.conv3_1x1(x)
        # x = self.model.gelu(x)
        # activations = x
        # return [activations]
    
"""
class featExtract_pretrained_RoBERTa(nn.Module):
    "
    #     TODO: Lets do this kinda later.
    This is a wrapper model used to extract features from the pretrained RoBERTa model.
    "
    def __init__(self, hf_cache_dir):
        "
        Args:
            hf_cache_dir (str): Path to the cache directory for the huggingface transformers
        "
        super(featExtract_pretrained_RoBERTa, self).__init__()
        self.RoBERTa = RobertaModel.from_pretrained("roberta-base", cache_dir=hf_cache_dir)

    def forward(self, feat, mask):
        "
        Args:
            feat (list): List of tensors with token-ids of shape (batch_size, seq_len, feat_dim)
            mask (list): List of tensors of shape (batch_size, seq_len)

        Returns:
            pooler_output (tensor): Tensor of shape (batch_size, 768)
        "
        x = self.RoBERTa(feat, mask)
        return x.pooler_output

"""