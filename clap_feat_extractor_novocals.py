import os
import torchaudio
from transformers import ClapModel, ClapProcessor
import torch
import pickle as pkl
def load_model(model_name):
    model = ClapModel.from_pretrained(model_name).to(0)
    processor = ClapProcessor.from_pretrained(model_name)
    return model, processor

def get_waveform(audio_name, folder):
    waveform = None
    for files in os.listdir(folder):
        if audio_name in files:
            if waveform is None:
                waveform, sample_rate = torchaudio.load(os.path.join(folder, files))
            else:
            #    append the waveform
                waveform = torch.cat((waveform, torchaudio.load(os.path.join(folder, files))[0]), dim=1)
    waveform = waveform.mean(0, keepdim=True)
    waveform = waveform.squeeze(0)
    return waveform

def get_feature(waveform, model, processor):
    # do feat extraction for every 10 seconds i.e., 160000 samples
    feature = None
    for i in range(0, waveform.shape[0], 160000):
        inputs = processor(audios=waveform[i:i+160000], return_tensors="pt").to(device="cuda:0",)
        audio_embed = model.get_audio_features(**inputs)
    # feature (audio_embed.last_hidden_state.squeeze(0)[:,:,30])
        # feature = torch.cat((feature, audio_embed.last_hidden_state.squeeze(0)[:,:,30]), dim=0)
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
    models_list = ["laion/larger_clap_general","laion/larger_clap_music_and_speech","laion/larger_clap_music","laion/clap-htsat-fused"]
    audios_list = ["no_vocals", "vocals","sfx","music","total"]
    for model_name in models_list:
        for audio_type in audios_list:
            base_audios_folder = f"/ssd_scratch/cvit/kolubex/data/audios/{audio_type}/{model_name.split('/')[1]}"
            base_feats_folder = f"/ssd_scratch/cvit/kolubex/data/audio_feats/{audio_type}/{model_name.split('/')[1]}"
            model, processor = load_model(model_name)
            os.makedirs(base_feats_folder, exist_ok=True)
            for movie in os.listdir(base_audios_folder):
                movie_folder = os.path.join(base_audios_folder, movie)
                for audio in os.listdir(movie_folder):
                    if "chunk1" in audio:
                        waveform = get_waveform(audio[:-11], movie_folder)
                        feature = get_feature(waveform, model, processor)
                        feature = feature.detach().cpu()
                        duration = waveform.shape[0]/16000
                        save_feature(audio[:-11], feature, duration, movie, base_feats_folder)
                        break
                break