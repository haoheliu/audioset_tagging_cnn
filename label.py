import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import os
import numpy as np 

PATH = "/vol/research/dcase2022/datasets/bbc_sound_effect/wav/-nhupart0"

def save(tensor, fname): 
    array = tensor.cpu().numpy() 
    np.save(fname, array) 
    
from tqdm import tqdm

for file in tqdm(os.listdir(PATH)):
    # import ipdb; ipdb.set_trace()
    audio_path = os.path.join(PATH, file)
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True, duration = 10)
    audio = audio[None, :]  # (batch_size, segment_samples)

    print('------ Audio tagging ------')
    at = AudioTagging(checkpoint_path=None, device='cuda')
    print(audio.shape)
    (clipwise_output, embedding) = at.inference(audio)
    save(clipwise_output, "%s_%s" % (file[:-4], "clipwise_output.npy"))
    save(embedding, "%s_%s" % (file[:-4], "embedding.npy"))

    print('------ Sound event detection ------')
    sed = SoundEventDetection(checkpoint_path=None, device='cuda')
    framewise_output = sed.inference(audio)
    
    