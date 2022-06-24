from sklearn import metrics
import numpy
import sklearn.metrics
from yaml import load
from pytorch_utils import forward
from mAP import *
import pickle
import os
from tqdm import tqdm

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
        self.sample_weight = {}
        self.weight = load_pickle("/vol/research/dcase2022/project/hhl_scripts/datasets/1_audioset/name_last_description_average_embedding.pkl")

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

        # Forward
        # output_dict = forward(
        #     model=self.model, 
        #     generator=data_loader, 
        #     return_target=True)
        
        output_dict = load_pickle("/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/audioset_tagging_cnn/output_dict.pkl")
        # save_pickle(output_dict, fname="output_dict.pkl")

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
        return np.array(ret)

    def get_sample_weight(self, target, class_weight):
        weight = []
        for i in range(target.shape[0]):
          weight.append(np.mean(class_weight[target[i,...]==1]))
        return weight

    def build_sample_weight(self, target):
        from tqdm import tqdm
        for i in tqdm(range(target.shape[1])):
          class_weight = self.get_weight(i, target.shape[1])
          self.sample_weight[i] = self.get_sample_weight(target, class_weight)
        save_pickle(self.sample_weight, "sample_weight.pkl")


    def random_choice(self, prob=0.5):
        coin = np.random.uniform(0,1)
        if(coin < prob): return True
        else: return False

    def drop_labels_by_prob(self, target, prob=0.5):
        for i in range(target.shape[0]):
          true_index = np.where(target[i] == 1)
          if(np.sum(target[i]) > 1 and self.random_choice(prob)):
            # import ipdb; ipdb.set_trace()
            target[i, np.random.choice(true_index[0])] = 0
        return target

    def evaluate_debug(self, prob=0.5):
        # Forward
        output_dict = load_pickle("/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/audioset_tagging_cnn/output_dict.pkl")
        
        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        # target = self.drop_labels_by_prob(target, prob)

        if(os.path.exists("sample_weight.pkl")):
          self.sample_weight = load_pickle("sample_weight.pkl")
        else:
          self.build_sample_weight(target)

        ap = []
        
        for i in tqdm(range(target.shape[1])):
          # sample_weight = self.sample_weight[i]
          sample_weight = None
          ap.append(_average_precision(target[:,i], clipwise_output[:,i], sample_weight=sample_weight))

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        
        # self.inspect_result(clipwise_output, target)

        statistics = {'average_precision': np.array(ap), 'auc': auc}

        return statistics

    def inspect_result(self, clipwise_output, target):
        import matplotlib.pyplot as plt
        for i in range(target.shape[1]):
          index = target[:, i] == 1
          target_class_dist = target[index]
          output_class_dist = clipwise_output[index]
          false_positive_class_dist = output_class_dist * (target_class_dist != 1)
          false_positive_class_dist = np.sum(false_positive_class_dist, axis=0)
          output_class_dist = np.sum(output_class_dist, axis=0)
      
          # diff = output_class_dist - false_positive_class_dist

          print("@"+self.label2class[i])
          wrongs =  [(self.label2class[x], false_positive_class_dist[x]) for x in np.argsort(false_positive_class_dist)[::-1][:30]]
          for each in wrongs:
            print("\t", each)

          # print("@"+self.label2class[i])
          # wrongs =  [(self.label2class[x], diff[x]) for x in np.argsort(diff)[::-1][:30]]
          # for each in wrongs:
          #   print("\t", each)

          # plt.plot(false_positive_class_dist)
          # plt.savefig("%s.png" % i)
          # plt.close()

if __name__ == "__main__":
  import numpy as np

  evaluator = Evaluator(None)
  res = evaluator.evaluate_debug()
  
  # for i in range(res['average_precision'].shape[0]):
  #   print(evaluator.label2class[i],"\t" ,res['average_precision'][i])

  for k in res.keys():
    print(k, res[k].shape, np.mean(res[k]))