import numpy as np
import pickle
import os
import glob
from tqdm import tqdm

PATH = "/vol/research/ai4sound/datasets/bbc_sound_effect/mp3"
audio_files = glob.glob(os.path.join(PATH,"*/*.mp3"))
print("Total files:", len(audio_files))


def write_file(string, fname):
    f = open(fname, "w")
    f.write(string)
    f.close()

def load_pickle(fname):
    # print("Load pickle at "+fname)
    with open(fname,'rb') as f:
        res = pickle.load(f)
    return res

def build_tagging(fname, tag_dict):
    ret = fname+","
    for k in tag_dict.keys():
        ret += "%s,%.4f," % (k.replace(",",";"), tag_dict[k])
    ret += "\n"
    return ret



tagging_row = ""

for idx, file in enumerate(tqdm(audio_files)):
    try:
        tagging_file = load_pickle(file.replace(".mp3","_tagging.pkl"))
        tagging_row += build_tagging(os.path.join(*file.split("/")[-3:]), tagging_file)
    except:
        continue
    # sed_file = load_pickle(file.replace(".mp3","_sed.pkl"))
    # break
    
write_file(tagging_row, "tagging_full.csv")
    