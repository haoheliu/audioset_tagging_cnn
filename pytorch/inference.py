import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config

at_model = None
sed_model = None
COUNTER=0
import pickle

def save_pickle(obj,fname):
    # print("Save pickle at "+fname)
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

def load_pickle(fname):
    # print("Load pickle at "+fname)
    with open(fname,'rb') as f:
        res = pickle.load(f)
    return res

def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """
    global at_model
    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    c_path = audio_path.replace(".wav","_tagging.pkl")
    
    if(os.path.exists(c_path)): return None, None
    
    if(at_model is None):
        # Model
        Model = eval(model_type)
        model = Model(sample_rate=sample_rate, window_size=window_size, 
            hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
            classes_num=classes_num)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

        # Parallel
        if 'cuda' in str(device):
            model.to(device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        else:
            print('Using CPU.')
        at_model = model
    else:
        model = at_model
    
    # Load audio
    if(not os.path.exists(audio_path)): return None, None
    
    try:
        (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(e, audio_path)
        return None, None

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    try:
        # Forward
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(waveform, None)
    except:
        print("Caught exception")
        return None, None

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    result = {}
    for k in range(50):
        a,b = np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]]
        print('{}: {:.3f}'.format(a,b))
        result[a] = b

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    # c_path = audio_path.replace(".wav","clipwise_output_tagging.npy")
    # label_path = audio_path.replace(".wav","labels_tagging.npy")
    # np.save(c_path, clipwise_output)
    # np.save(label_path, labels)
    save_pickle(result, c_path)
    return clipwise_output, labels

def sound_event_detection(args):
    """Inference sound event detection result of an audio clip.
    """
    global sed_model
    global COUNTER

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    c_path = audio_path.replace(".wav","_sed.pkl")
    # Paths
    fig_path = os.path.join('/vol/research/ai4sound/datasets/bbc_sound_effect/sed_pics', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))
    
    try:
        (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(e, audio_path)
        return None, None

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)
    top_k = 10  # Show top results
    if(not os.path.exists(c_path)): 
        if(sed_model is None):
            # Model
            Model = eval(model_type)
            model = Model(sample_rate=sample_rate, window_size=window_size, 
                hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                classes_num=classes_num)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])

            # Parallel
            print('GPU number: {}'.format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)

            if 'cuda' in str(device):
                model.to(device)
            sed_model = model
        else:
            model = sed_model
        
        # Load audio
        # if(not os.path.exists(audio_path)): return None, None

        # Forward
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(waveform, None)

        framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
        """(time_steps, classes_num)"""

        print('Sound event detection result (time_steps x classes_num): {}'.format(
            framewise_output.shape))

        sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

        result = {}
        
        result['top_result_mat'] = framewise_output[:, sorted_indexes[0 : top_k]].T  
        result['top_labels'] = np.array(labels)[sorted_indexes[0 : top_k]] 
        save_pickle(result, c_path)
        return framewise_output, labels
    else:
        if(os.path.exists(fig_path)): 
            COUNTER += 1
            return None, None
        print("draw", c_path)
        # Plot result    
        fig_path = os.path.join(os.path.dirname(fig_path),os.path.basename(os.path.dirname(c_path))+"_"+os.path.basename(fig_path))
        result = load_pickle(c_path)
        stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
            hop_length=hop_size, window='hann', center=True)
        frames_num = stft.shape[-1]

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
        axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
        axs[0].set_ylabel('Frequency bins')
        axs[0].set_title('Log spectrogram')
        axs[1].matshow(result['top_result_mat'], origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_num/10))
        axs[1].xaxis.set_ticklabels(np.around(np.arange(0, frames_num / frames_per_second, (frames_num / frames_per_second)/10), 1))
        axs[1].yaxis.set_ticks(np.arange(0, top_k))
        axs[1].yaxis.set_ticklabels(result['top_labels'][:top_k])
        axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        axs[1].set_xlabel('Seconds')
        axs[1].xaxis.set_ticks_position('bottom')

        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print('Save sound event detection visualization to {}'.format(fig_path))
        return None, None
    

def recursive_glob(path, suffix):
    from glob import glob
    return glob(os.path.join(path,"*" + suffix)) + \
                glob(os.path.join(path,"*/*" + suffix)) + \
                    glob(os.path.join(path,"*/*/*" + suffix)) + \
                        glob(os.path.join(path,"*/*/*/*" + suffix)) + \
                            glob(os.path.join(path,"*/*/*/*/*" + suffix)) + \
                                glob(os.path.join(path,"*/*/*/*/*/*" + suffix))
        
        
def mp3_to_wav(src, dst):
    import os
    from os import path
    if(".wav" in src): return src, False
    else:
        dst = src.replace(".mp3",".wav")
    # from pydub import AudioSegment  
    # sound = AudioSegment.from_mp3(src)
    # sound.export(dst, format="wav")
    cmd = "sox %s %s" % (src, dst)
    os.system(cmd) # TODO
    return dst, True
    
if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000) 
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=False)
    parser_at.add_argument('--cuda', action='store_true', default=False)
    parser_at.add_argument('--path', required=True)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000) 
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--audio_path', type=str, required=False)
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    parser_sed.add_argument('--path', required=True)
    
    args = parser.parse_args()
    from tqdm import tqdm
    PATH = args.path # "/vol/research/ai4sound/datasets/bbc_sound_effect/"
    print(PATH, args.path)
    print(len(list(recursive_glob(PATH,".wav")+recursive_glob(PATH,".mp3"))))
    
    for file in tqdm(np.random.permutation(list(recursive_glob(PATH,"*.wav")+recursive_glob(PATH,"*.mp3")))):
        if(".wav" not in file and ".mp3" not in file): continue
        
        if args.mode == 'audio_tagging':
            if(os.path.exists(file.replace(".mp3","_tagging.pkl"))): continue
        elif args.mode == 'sound_event_detection':
            # if(os.path.exists(file.replace(".mp3","_sed.pkl"))): 
            #     continue
            pass
        
        # TODO
        fig_path = os.path.join('/vol/research/ai4sound/datasets/bbc_sound_effect/sed_pics', '{}.png'.format(get_filename(file)))
        fig_path = os.path.join(os.path.dirname(fig_path),os.path.basename(os.path.dirname(file))+"_"+os.path.basename(fig_path))
        if(os.path.exists(fig_path)): continue
        ################################   
        
        file, converted = mp3_to_wav(file, file)
        args.audio_path = os.path.join(PATH, file)
        
        if(not os.path.exists(args.audio_path)): continue
        
        if args.mode == 'audio_tagging':
            audio_tagging(args)
        elif args.mode == 'sound_event_detection':
            sound_event_detection(args)
        
        if(converted):
            try:
                os.remove(file)
            except Exception as e:
                print(e, file)
    print(COUNTER)