from df import enhance, init_df
import os
from natsort import natsorted
import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa
import torchaudio
from tools.compute_metrics import compute_metrics

#等我48000sr的VBD再測
if __name__ == "__main__":

    model_name = "DeepFilterNet2"
    model, df_state, _ = init_df(model_name)  # Load default model

    clean_dir = "./clean_testset_wav"
    #noise_dir = "../VoiceBank_Demand_48000/noisy_testset_wav"
    enh_dir = f'./enhanced_{model_name}'
    if not os.path.exists(enh_dir):
        os.mkdir(enh_dir)

    audio_list = os.listdir(clean_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(6)

    # for audio in tqdm(audio_list):
    #     noisy_path = os.path.join(noise_dir, audio)
    #     noisy, sr = torchaudio.load(noisy_path)
    #     enhanced_audio = enhance(model, df_state, noisy)
    #     saved_path = os.path.join(enh_dir, audio)
    #     torchaudio.save(saved_path, enhanced_audio, sr)

    for audio in tqdm(audio_list):
        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr = sf.read(clean_path)
        # if sr != 16000:
        #     clean_audio = librosa.resample(clean_audio, orig_sr=sr, target_sr=16000)
        enh_path = os.path.join(enh_dir , audio)
        enh_audio, sr = sf.read(enh_path)
        # if sr != 16000:
        #     enh_audio = librosa.resample(enh_audio, orig_sr=sr, target_sr=16000)
        metrics = compute_metrics(clean_audio, enh_audio, 16000, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print(
        "pesq: ",
        metrics_avg[0],
        "csig: ",
        metrics_avg[1],
        "cbak: ",
        metrics_avg[2],
        "covl: ",
        metrics_avg[3],
        "ssnr: ",
        metrics_avg[4],
        "stoi: ",
        metrics_avg[5],
    )
