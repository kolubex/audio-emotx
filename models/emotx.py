import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FramePositionalEncoding(nn.Module):
    """
    Fourier positional encoding from "Attention is all you need"
    """
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        """
        Args:
            d_model (int): Dimension of the features
            dropout (float): Dropout rate
            max_len (int): Maximum index for the positional embedding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        # print("pe.shape: ", pe.shape)
        # print("max_len: ", max_len)
        # print("d_model: ", d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, feat, frames):
        """
        Args:
            feat (torch.Tensor): Feature vectors. Tensor of shape (batch_size, num_frames, feat_dim)
            frames (torch.Tensor): Frame indexes. Tensor of shape (batch_size, num_frames)

        Returns:
            torch.Tensor: Positional Embedding added to features. Tensor of shape (batch_size, num_frames, feat_dim)
        """
        # print("feat.shape: ", feat.shape)
        # print("frames.shape: ", frames.shape)
        pos_emb = self.pe[:, frames, :]
        out = feat + pos_emb.squeeze(0)
        return out # self.dropout(out)


class EmoTx(nn.Module):
    """
    Main EmoTx model proposed in "How you feelin'? Learning Emotions and Mental States in Movie Scenes. (CVPR23)"
    """
    def __init__(self, num_labels, num_pos_embeddings, scene_feat_dim, char_feat_dim, srt_feat_dim,audio_feat_dim, num_enc_layers=4, num_chars=8, max_individual_tokens=100, hidden_dim=512):
        """
        Args:
            num_labels (int): Number of labels.
            num_pos_embeddings (int): Maximum index for positional embeddings.
            scene_feat_dim (int): Dimension of scene features.
            char_feat_dim (int): Dimension of character features.
            srt_feat_dim (int): Dimension of utterance/subtitle features.
            num_enc_layers (int): Number of transformer encoder layers.
            num_chars (int): Number of characters.
            max_individual_tokens (int): Maximum number of tokens for scene, each character and subtitle features.
            hidden_dim (int): Dimension of hidden layers for projection of scene, character and subtitle features.
        """
        super(EmoTx, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_chars = num_chars
        self.max_individual_tokens = max_individual_tokens
        self.scene_dim_reduction = nn.Linear(scene_feat_dim, self.hidden_dim)
        self.char_dim_reduction = nn.Linear(char_feat_dim, self.hidden_dim)
        self.srt_dim_reduction = nn.Linear(srt_feat_dim, self.hidden_dim)
        self.audio_dim_reduction = nn.Linear(audio_feat_dim, self.hidden_dim)
        self.positional_encoder = FramePositionalEncoding(self.hidden_dim, 0.1, max_len=num_pos_embeddings)
        self.type_emb = nn.Embedding(4, self.hidden_dim)
        self.char_emb = nn.Embedding(self.num_chars, self.hidden_dim)
        self.cls_emb_srt = nn.Embedding(num_labels, self.hidden_dim)
        self.cls_emb_audio = nn.Embedding(num_labels, self.hidden_dim)
        self.cls_emb_scene = nn.Embedding(num_labels, self.hidden_dim)
        self.cls_emb_char = nn.Embedding(num_labels, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.logits = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for i in range(self.num_labels)])
        self.cls_mask = torch.zeros((1, self.num_labels))
        self.seq_len = (self.num_chars+3)*max_individual_tokens + (self.num_chars+1)*self.num_labels
        self.src_mask = torch.zeros((self.seq_len, self.seq_len)).to(torch.bool)

    def append_tokens(self, batch_size, feat, feat_cls_emb):
        """
        Append CLS tokens to the beginning of the feature vectors.

        Args:
            batch_size (int): Batch size.
            feat (torch.Tensor): Feature vectors. Tensor of shape (batch_size, num_frames, feat_dim)
            feat_cls_emb (torch.nn.Embedding): Embedding layer for CLS tokens.

        Returns:
            feat (torch.Tensor): Feature vectors with CLS tokens. Tensor of shape (batch_size, num_frames+num_labels, feat_dim)
        """
        cls_emb = torch.cat([feat_cls_emb(torch.tensor([i], device=feat.device)) for i in range(self.num_labels)], dim=0)
        cls_emb = cls_emb.repeat(batch_size, 1, 1)
        feat = torch.cat([cls_emb, feat], dim=1)
        return feat

    def forward(self, feats, masks):
        """
        Args:
            feats (list): List of feature vectors. It contains the following:
                char_feats (torch.Tensor): Character features. Tensor of shape (batch_size, num_chars, num_frames, feat_dim)
                scene_feat (torch.Tensor): Scene features. Tensor of shape (batch_size, num_frames, feat_dim)
                srt_feats (torch.Tensor): Utterance/subtitle features. Tensor of shape (batch_size, num_frames, feat_dim)
                audio_feats (torch.Tensor): Audio features. Tensor of shape (batch_size, num_frames, feat_dim)
                char_frames (torch.Tensor): Character frame indexes. Tensor of shape (batch_size, num_chars, num_frames)
                scene_frames (torch.Tensor): Scene frame indexes. Tensor of shape (batch_size, num_frames)
                srt_bins (torch.Tensor): Utterance/subtitle bin indexes. Tensor of shape (batch_size, num_frames)
                audio_bins (torch.Tensor): Audio bin indexes. Tensor of shape (batch_size, num_frames)
            masks (list): List of masks. It contains the following:
                char_feat_masks (torch.Tensor): Character feature masks. Tensor of shape (batch_size, num_chars, num_frames)
                scene_feat_masks (torch.Tensor): Scene feature masks. Tensor of shape (batch_size, num_frames)
                srt_mask (torch.Tensor): Utterance/subtitle mask. Tensor of shape (batch_size, num_frames)
                audio_mask (torch.Tensor): Audio mask. Tensor of shape (batch_size, num_frames)
                char_target_mask (torch.Tensor): Character target masks. Tensor of shape (batch_size, num_chars, num_labels)
                scene_target_mask (torch.Tensor): Scene target masks. Tensor of shape (batch_size, num_labels)

        Returns:
            filtered_pres (list): List of logits for each emotion label.
                The first element is the logits for the characters. Shape: (number_of_characters_across_batches, num_labels)
                The second element is the logits for the scene. Shape: (batch_size, num_labels)
        """

        # Unpacking the features, and masks from the arguments
        char_feats, scene_feat, srt_feats, audio_feats, char_frames, scene_frames, srt_bins, audio_bins = feats
        char_feat_masks, scene_feat_masks, srt_mask, audio_mask, char_target_mask, scene_target_mask = masks
        target_masks = [char_target_mask, scene_target_mask] # Will be used to filter the logits for masked characters.

        batch_size = char_feats.shape[0]

        # Project the scene, character and subtitle features to a common dimension.
        scene_feat = self.scene_dim_reduction(scene_feat)
        char_feats = self.char_dim_reduction(char_feats)
        srt_feats = self.srt_dim_reduction(srt_feats)
        audio_feats = self.audio_dim_reduction(audio_feats)

        # Add positional embeddings to scene and char features
        scene_feat = self.positional_encoder(scene_feat, scene_frames)
        char_feats = torch.stack([self.positional_encoder(char_feats[:,i,:,:], char_frames[:,i,:]) for i in range(self.num_chars)], dim=1)
        # print("srt_feats.shape: ", srt_feats.shape)
        # print("srt_bins.shape: ", srt_bins.shape)
        srt_feats = self.positional_encoder(srt_feats, srt_bins)
        # print(srt_bins)
        # print("audio_feats.shape: ", audio_feats.shape)
        # print("audio_bins.shape: ", audio_bins.shape)
        # print(audio_bins)
        audio_feats = self.positional_encoder(audio_feats, audio_bins)

        # Append CLS tokens around scene features
        scene_feat = self.append_tokens(batch_size, scene_feat, self.cls_emb_scene)

        # Append CLS tokens around character features
        char_feats = torch.stack([self.append_tokens(batch_size, char_feats[:,i,:,:], self.cls_emb_char) for i in range(self.num_chars)], dim=1)

        # Add type embeddings to scene, character and subtitle, audio features.
        scene_type_emb = self.type_emb(torch.tensor([0]).to(scene_feat.device)).repeat(batch_size, scene_feat.shape[1], 1)
        char_type_emb = self.type_emb(torch.tensor([1]).to(scene_feat.device)).repeat(batch_size, char_feats.shape[1], char_feats.shape[2], 1)
        srt_type_emb = self.type_emb(torch.tensor([2]).to(srt_feats.device)).repeat(batch_size, srt_feats.shape[1], 1)
        audio_type_emb = self.type_emb(torch.tensor([3]).to(audio_feats.device)).repeat(batch_size, audio_feats.shape[1], 1)
        scene_feat = scene_feat + scene_type_emb
        char_feats = char_feats + char_type_emb
        srt_feats = srt_feats + srt_type_emb
        audio_feats = audio_feats + audio_type_emb

        # Add character index embeddings to char features
        char_embs = self.char_emb(torch.tensor(list(range(self.num_chars))).to(char_feats.device))
        char_feats = torch.stack([char_feats[:, i,:,:] + char_embs[i].repeat(batch_size, char_feats.shape[-2], 1) for i in range(self.num_chars)], dim=1)

        # Update scene and char_feat masks to consider CLS as not masked token
        feat_mask = self.cls_mask.repeat(batch_size, 1).to(scene_feat.device)
        scene_mask = torch.cat([feat_mask, scene_feat_masks], dim=1)
        char_mask = torch.stack([torch.cat([feat_mask, char_feat_masks[:, i,:]], dim=1) for i in range(self.num_chars)], dim=1)

        # Stack all the characters together
        start_ndxs = torch.cumsum(torch.tensor([0]+[scene_feat.shape[1]]+[char_feats.shape[-2]]*(self.num_chars-1)), dim=0).to(char_feats.device)
        cls_ndxs = torch.cumsum(torch.tensor([[int(i)] + [1]*(self.num_labels-1) for i in start_ndxs]), dim=1).reshape(-1)
        char_feats = char_feats.flatten(1,2)
        char_mask = char_mask.flatten(1,2)

        # Concatenate scene, character and subtitle features and mask. This prepares the input for the encoder.
        feat = torch.cat([scene_feat, char_feats, srt_feats,audio_feats], dim=1)
        mask = torch.cat([scene_mask, char_mask, srt_mask, audio_mask], dim=1)
        # print("feat.shape: ", feat.shape)
        # print("scene_feat.shape: ", scene_feat.shape)
        # print("char_feats.shape: ", char_feats.shape)
        # print("srt_feats.shape: ", srt_feats.shape)
        # print("audio_feats.shape: ", audio_feats.shape)
        # print("mask.shape: ", mask.shape)
        # print("scene_mask.shape: ", scene_mask.shape)
        # print("char_mask.shape: ", char_mask.shape)
        # print("srt_mask.shape: ", srt_mask.shape)
        # print("audio_mask.shape: ", audio_mask.shape)

        # Normalize the feats after attaching all embeddings. Input to the transformer encoder is now ready.
        feat = self.layer_norm(feat)
        # make 1 in self.src_mask as True and 0 as False
        self.src_mask = self.src_mask.to(feat.device).bool()
        mask = mask.to(feat.device).bool()
        # Pass the prepared input to transformer encoder
        out1 = self.encoder.layers[0](feat,src_mask=self.src_mask.to(feat.device), src_key_padding_mask=mask)
        out = self.encoder.layers[0].self_attn(feat,feat,feat, attn_mask=self.src_mask.to(feat.device), key_padding_mask=mask)[1]
        # save attn1 as a numpy array
        # torch.save(out.detach().cpu(), '/ssd_scratch/cvit/kolubex/checkpoints/metadata/attnplots/attn1.pt')
        # torch.save(char_frames.detach().cpu(), '/ssd_scratch/cvit/kolubex/checkpoints/metadata/attnplots/char_bins.pt')
        # torch.save(scene_frames.detach().cpu(), '/ssd_scratch/cvit/kolubex/checkpoints/metadata/attnplots/scene_bins.pt')
        # torch.save(srt_bins.detach().cpu(), '/ssd_scratch/cvit/kolubex/checkpoints/metadata/attnplots/srt_bins.pt')
        # torch.save(audio_bins.detach().cpu(), '/ssd_scratch/cvit/kolubex/checkpoints/metadata/attnplots/audio_bins.pt')
        out = self.encoder.layers[1].self_attn(out1, out1, out1, attn_mask=self.src_mask.to(feat.device), key_padding_mask=mask)[1]
        # save attn2 as a .pt file
        # torch.save(out.detach().cpu(), '/ssd_scratch/cvit/kolubex/checkpoints/metadata/attnplots/attn.pt')
        # torch.save(mask.detach().cpu(), '/ssd_scratch/cvit/kolubex/checkpoints/metadata/attnplots/mask.pt')
        # torch.save(cls_ndxs.detach().cpu(), '/ssd_scratch/cvit/kolubex/checkpoints/metadata/attnplots/cls_ndxs.pt')
        out1 = out1.detach().cpu().numpy()
        del out1
        out = self.encoder(feat, mask=self.src_mask.to(feat.device), src_key_padding_mask=mask)
        cls_heads = out[:, cls_ndxs, :]
        cls_heads = cls_heads.reshape(-1, self.num_chars+1, self.num_labels, self.hidden_dim)

        # Pass the CLS embeddings to respective logits layer to get the logit for each label.
        # These individual logits are concatenated to generate the final logits.
        preds = torch.empty((batch_size, self.num_chars+1, 0)).to(feat.device)
        for ndx in range(self.num_labels):
            logit = self.logits[ndx](cls_heads[:, :, ndx, :])
            preds = torch.cat([preds, logit], dim=2)

        # Filter the logits for masked characters.
        # filtered_preds[0] is the logits for the characters. (character_counts_across_batches, num_labels)
        # filtered_preds[1] is the logits for the scene. Shape will be (batch_size, num_labels)
        filtered_preds = [preds.flatten()[mask] for mask in target_masks]
        filtered_preds = [filtered_pred.reshape(-1, self.num_labels) for filtered_pred in filtered_preds]

        return filtered_preds
