from sklearn import metrics
import numpy
import sklearn.metrics
from yaml import load
from pytorch_utils import forward
from mAP import *
import pickle
import os
from tqdm import tqdm

EMP_ALPHA = 0.05

def save_pickle(obj,fname):
    print("Save pickle at "+fname)
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

def load_pickle(fname):
    print("Load pickle at "+fname)
    with open(fname,'rb') as f:
        res = pickle.load(f)
    return res

def _average_precision(y_true, pred_scores, sample_weight = None):
    precisions, recalls, thresholds = precision_recall_curve(y_true, pred_scores, sample_weight=sample_weight)
    precisions = numpy.array(precisions)
    recalls = numpy.array(recalls)
    AP = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    return AP

class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        self.label2class = {}
        self.build_label_to_class()
        self.sample_weight = None
        self.weight = np.load("/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/audioset_tagging_cnn/pytorch/audioset_co_occurance.npy")


    def build_label_to_class(self):
        import pandas as pd
        csv = pd.read_csv("/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/audioset_tagging_cnn/datasets/audioset201906/metadata/class_labels_indices.csv")
        for i, row in csv.iterrows():
          self.label2class[int(row['index'])] = row['display_name']
        
    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """
        
        output_dict = load_pickle("/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/audioset_tagging_cnn/output_dict_test.pkl")
        # save_pickle(output_dict, fname="output_dict.pkl")
        index = np.sum(output_dict['target'] ,axis=1) != 0
        for k in output_dict.keys():
          output_dict[k] = output_dict[k][index]
        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)
        
        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        
        statistics = {'average_precision': average_precision, 'auc': auc}

        return statistics

    def get_weight(self, class_idx, class_num):
        ret = []
        for i in range(class_num):
          if(class_idx == i): 
            ret.append(0.5)
            continue
          ret.append(self.weight[self.label2class[class_idx]][self.label2class[i]])
          # ret.append(0.5)
        return np.array(ret)

    def get_sample_weight(self, target, class_weight):
        weight = []
        for i in range(target.shape[0]):
          weight.append(np.min(class_weight[target[i,...]==1]))
        weight = np.array(weight)
        return weight

    def reset_sample_weight(self):
        os.remove("sample_weight.pkl")

    def build_sample_weight(self, target):
        # for i in tqdm(range(target.shape[1])):
            # pass
            # Original setting: In class i, false positive is 100% counted in this sample.
            # self.sample_weight[i] = 1-target[:, i] 
        # New setting: In class i, calculate the counting weight for each sample
        for counter, j in tqdm(enumerate(range(target.shape[0]))): 
            # The label of j-th audiofile
            labels = target[j,:]
            # Get the co-occurance distribution for each positive label
            class_freq_distribution = self.weight[labels==1,:]
            # Ignore the positive labels when calculating false positive weight
            class_freq_distribution[:, labels==1] *= 0.0
            # Normalize the dist of each label
            # class_freq_distribution = class_freq_distribution / max(np.max(class_freq_distribution, axis=1, keepdims=True), 1)
            max_freq = np.max(class_freq_distribution, axis=1, keepdims=True)
            max_freq = np.clip(max_freq, a_min=1, a_max=None)
            class_freq_distribution = class_freq_distribution / max_freq
            # Sum the co-occurance of the labels
            class_freq_distribution = np.mean(class_freq_distribution, axis=0)
            # Remove the co-occurance data of the positive labels, we are only interested in the false negative in the sample
            self.sample_weight[j] = class_freq_distribution

            assert np.isnan(class_freq_distribution).any() == False, max_freq

            # if(counter < 30):
            #   import matplotlib.pyplot as plt
            #   positive_label = str([self.label2class[x] for x in np.where(labels==1)[0]])
            #   plt.figure(figsize=(10, 3))
            #   plt.plot(class_freq_distribution)
            #   plt.savefig("%s.png" % (positive_label))
            #   plt.close()

        save_pickle(self.sample_weight, "sample_weight.pkl")
            
    def evaluate_debug(self, prob=0.0):
        # Forward
        output_dict = load_pickle("/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/audioset_tagging_cnn/output_dict_test.pkl")
        index = np.sum(output_dict['target'] ,axis=1) != 0
        for k in output_dict.keys():
          output_dict[k] = output_dict[k][index]

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)
        # import ipdb; ipdb.set_trace()
        if(self.sample_weight is None):
          # self.sample_weight[i] store the false positive penalty for each sample (if this sample have false positive of class i)
          self.sample_weight = {}
          self.build_sample_weight(target)
        
        ap=[]
        sim_ap = []
        dis_sim_ap=[]
        
        for i in tqdm(range(target.shape[1])):
          # sample_weight = None
          weight = []
          for x in self.sample_weight.keys():
              weight.append(self.sample_weight[x][i])
          weight = np.array(weight)
          # Weight 0-1: 
          #   if closer to 1, not expect to be false positive (high correlation)
          #   if closer to 0, expect to be false positive (low correlation)
          weight = weight**0.25
          ap.append(_average_precision(target[:,i], clipwise_output[:,i], None))
          dis_sim_ap.append(_average_precision(target[:,i], clipwise_output[:,i], 1-weight)) # Ignore the sound that most likely to co-appear
          sim_ap.append(_average_precision(target[:,i], clipwise_output[:,i], 1+weight)) # Address the sound that most likely to co-appear

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(20, 5))
        # plt.subplot(311)
        # plt.plot(ap)
        # plt.subplot(312)
        # plt.plot(np.array(ap)-np.array(sim_ap))
        # plt.subplot(313)
        # plt.plot(np.array(ap)-np.array(dis_sim_ap))
        # plt.savefig("temp.png")
        # plt.close()
        auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        statistics = {'average_precision': np.array(ap), 'auc': auc, 'dis_sim_mAP': np.array(dis_sim_ap), 'sim_mAP': np.array(sim_ap)}
        return statistics

if __name__ == "__main__":
  import numpy as np
  
  evaluator = Evaluator(None)
  res = evaluator.evaluate_debug()
  for k in res.keys():
    print(k, np.mean(res[k]))