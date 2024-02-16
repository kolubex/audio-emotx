import os
import torchaudio
from transformers import ClapModel, ClapProcessor
import torch
import pickle as pkl
import argparse

def load_model(model_name, device):
    model = ClapModel.from_pretrained(model_name).to(device)
    processor = ClapProcessor.from_pretrained(model_name)
    return model, processor

def get_waveform(audio_name, folder):
    waveform = None
    for files in os.listdir(folder):
        if audio_name in files:
            if waveform is None:
                waveform, sample_rate = torchaudio.load(os.path.join(folder, files))
            else:
                # append the waveform
                waveform = torch.cat((waveform, torchaudio.load(os.path.join(folder, files))[0]), dim=1)
    waveform = waveform.mean(0, keepdim=True)
    waveform = waveform.squeeze(0)
    return waveform

def get_feature(waveform, model, processor, device):
    feature = None
    for i in range(0, waveform.shape[0], 160000):
        inputs = processor(audios=waveform[i:i+160000], return_tensors="pt").to(device)
        audio_embed = model.get_audio_features(**inputs)
        if feature is None:
            feature = audio_embed.last_hidden_state.squeeze(0)[:,:,:30]
        else:
            feature = torch.cat((feature, audio_embed.last_hidden_state.squeeze(0)[:,:,:30]), dim=2)
    return feature

def save_feature(audio_name, feature, duration, folder, base_folder):
    os.makedirs(os.path.join(base_folder, folder), exist_ok=True)
    with open(os.path.join(base_folder, folder, audio_name + ".pkl"), 'wb') as f:
        pkl.dump(duration, f)
        pkl.dump(feature, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Extraction Script")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--audio_type", type=str, default="no_vocals", help="Type of audio data")
    args = parser.parse_args()

    models_list = ["laion/larger_clap_general", "laion/larger_clap_music_and_speech", "laion/larger_clap_music", "laion/clap-htsat-fused"]
    for model_name in models_list:
        base_audios_folder = f"/ssd_scratch/cvit/kolubex/data/audios/{args.audio_type}"
        base_feats_folder = f"/ssd_scratch/cvit/kolubex/data/audio_feats/{args.audio_type}/{model_name.split('/')[1]}"
        model, processor = load_model(model_name, f"cuda:{args.gpu}")
        os.makedirs(base_feats_folder, exist_ok=True)
        for movie in os.listdir(base_audios_folder):
            movie_folder = os.path.join(base_audios_folder, movie)
            for audio in os.listdir(movie_folder):
                if "chunk1" in audio:
                    waveform = get_waveform(audio[:-11], movie_folder)
                    feature = get_feature(waveform, model, processor, f"cuda:{args.gpu}")
                    feature = feature.detach().cpu()
                    duration = waveform.shape[0]/16000
                    save_feature(audio[:-11], feature, duration, movie, base_feats_folder)
                    # break
            # break
