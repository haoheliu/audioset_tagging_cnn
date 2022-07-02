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

    def get_sample_weight(self, target, class_weight, act_func = lambda x:x):
        weight = []
        for i in range(target.shape[0]):
          weight.append(np.min(class_weight[target[i,...]==1]))
        weight = np.array(weight)
        weight = act_func(weight)
        return weight

    def act_power_4(self, x):
        return x**4

    def act_power_3(self, x):
        return x**3

    def act_power_2(self, x):
        return x**2

    def act_power_1(self, x):
        return x

    def act_power_0_5(self, x):
        return x**0.5

    def act_power_0_25(self, x):
        return x**0.25

    def emp_near(self, x):
        # 0.5: [0.6114750189829752, 0.6067263970681768, 0.602657668778519, 0.595789289160237, 0.5921017546921142, 0.5865309491814862, 0.5812253104901342, 0.5753627485274455, 0.5702593752499303, 0.5659299603531596, 0.5584960575688561, 0.5522362563813357, 0.5494763758657324, 0.5397904162232039, 0.5354803903710866, 0.5259134110137785, 0.5184340042913396, 0.5125574565130243, 0.5024393492936859, 0.5018704775209525]
        # Mostly using the dis-similar labels for the evalution of the model
        x[x>EMP_ALPHA] = 1.0
        x[x<EMP_ALPHA] = 1e-4
        return x
      
    def emp_far(self, x):
        # Only use the similar labels for evaluation of the model (fine-grain classification)
        x[x>EMP_ALPHA] = 1e-4
        x[x<EMP_ALPHA] = 1.0
        return x

    def build_sample_weight(self, target, act_func = lambda x:x):
        from tqdm import tqdm
        for i in tqdm(range(target.shape[1])):
          class_weight = self.get_weight(i, target.shape[1])
          self.sample_weight[i] = self.get_sample_weight(target, class_weight, act_func)
        save_pickle(self.sample_weight, "sample_weight.pkl")

    def random_choice(self, prob=0.5):
        coin = np.random.uniform(0,1)
        if(coin < prob): return True
        else: return False

    def drop_labels_by_prob(self, target, prob=0.0):
        if(prob == 0):
          return target
        for i in range(target.shape[0]):
          true_index = np.where(target[i] == 1)
          # import ipdb; ipdb.set_trace()
          for j, each in enumerate(true_index[0]):
            # if(j == 0): continue
            if(self.random_choice(prob)):
              target[i, each] = 0
        return target

    def reset_sample_weight(self):
        os.remove("sample_weight.pkl")

    def evaluate_debug(self, prob=0.0, act_func = lambda x:x):
        # Forward
        output_dict = load_pickle("/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/audioset_tagging_cnn/output_dict_test.pkl")
        index = np.sum(output_dict['target'] ,axis=1) != 0
        for k in output_dict.keys():
          output_dict[k] = output_dict[k][index]

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        if(os.path.exists("sample_weight.pkl")):
          self.sample_weight = load_pickle("sample_weight.pkl")
        else:
          self.build_sample_weight(target, act_func)
        
        target = self.drop_labels_by_prob(target, prob=prob)

        ap = []
        
        for i in tqdm(range(target.shape[1])):
          sample_weight = [x for x in self.sample_weight[i]]
          # sample_weight = None
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

def mean_emp_near():
  # [0.432900907095631, 0.4138473229028886, 0.39471204816396727, 0.3930629978232302, 0.39489679305035436, 0.40156757973381424, 0.40516095473857183, 0.4135338917834924, 0.43040165773490985, 0.4505962606838141, 0.4915645574248436, 0.5619195543148767, 0.6037597557017337, 0.646282238426039, 0.6945003934919138, 0.7505016137991578, 0.8155607555745495, 0.8712949195135596] 0.5314480112198526
  global EMP_ALPHA
  import numpy as np
  res = {}
  evaluator = Evaluator(None)
  act_func = [evaluator.emp_near] # evaluator.act_power_0_25, evaluator.act_power_0_5, evaluator.act_power_1, evaluator.act_power_2, evaluator.act_power_3, evaluator.act_power_4
  
  for func in act_func:
    mAP = []
    for i in np.arange(0.0,0.9,0.05):
      EMP_ALPHA = i
      res = evaluator.evaluate_debug(prob=i, act_func=func)
      mAP.append(np.mean(res['average_precision']))
      evaluator.reset_sample_weight()
      print(mAP)
    print(mAP, np.mean(mAP))
    res[func] = mAP
    
def mean_emp_far():
  global EMP_ALPHA
  import numpy as np
  res = {}
  evaluator = Evaluator(None)
  act_func = [evaluator.emp_near] # evaluator.act_power_0_25, evaluator.act_power_0_5, evaluator.act_power_1, evaluator.act_power_2, evaluator.act_power_3, evaluator.act_power_4
  
  for func in act_func:
    mAP = []
    for i in np.arange(0.0,0.9,0.05):
      EMP_ALPHA = i
      res = evaluator.evaluate_debug(prob=i, act_func=func)
      mAP.append(np.mean(res['average_precision']))
      evaluator.reset_sample_weight()
      print(mAP)
    print(mAP, np.mean(mAP))
    res[func] = mAP

def robustness():
  import numpy as np
  res = {}
  evaluator = Evaluator(None)
  act_func = [evaluator.emp_near] # evaluator.act_power_0_25, evaluator.act_power_0_5, evaluator.act_power_1, evaluator.act_power_2, evaluator.act_power_3, evaluator.act_power_4
  
  for func in act_func:
    mAP = []
    for i in np.arange(0.0,0.5,0.025):
      res = evaluator.evaluate_debug(prob=i, act_func=func)
      mAP.append(np.mean(res['average_precision']))
    print(mAP)
    res[func] = mAP
    evaluator.reset_sample_weight()

  def minus(arr, x):
    return [y-x for y in arr]

  x_0_25 = [0.37160083010873424, 0.36091106087754793, 0.3495388830333401, 0.3381241583296012, 0.3314143780781964, 0.3233194896369214, 0.3101675299129076, 0.3019588324448509, 0.28954472536557724, 0.28246235078170806, 0.2738827370359669, 0.2635168254859729, 0.2517731374303718, 0.24300735610891896, 0.23644451335481856, 0.22655410865710426, 0.21927480098692734, 0.20893077978476704, 0.1985122210587024, 0.1928848048921635]
  x_0_5 = [0.40373917865451486, 0.39426142840601514, 0.384612531987614, 0.37558397921429343, 0.3647451894005105, 0.35316573560266734, 0.34734767357975377, 0.338607359408028, 0.3250967558046388, 0.31863741081484803, 0.3103342487651985, 0.29934122812330155, 0.28723188898937146, 0.28352280768734633, 0.2721614180125716, 0.2616941033233928, 0.2525943564822391, 0.24784864662019893, 0.2332880514735369, 0.22468574189630555]
  x_0 = [0.432900907095631, 0.4231023013207829, 0.414287189683723, 0.40454217644558366, 0.39364284877804906, 0.3819869252438288, 0.3729258567445653, 0.3629487768047105, 0.3590595628007568, 0.34627947080899085, 0.33496815424893633, 0.32546661825804807, 0.31962178406152564, 0.30361389350026585, 0.29675940612337165, 0.28856956511667026, 0.27973243721673996, 0.2677254505866966, 0.2600797568767678, 0.24448926641549673]
  x_1 = [0.4646550088123959, 0.45581658383971735, 0.44798244563658424, 0.4430335860779526, 0.4323502773422413, 0.42208987237419127, 0.4146141572734403, 0.4079865426139327, 0.39610477496946295, 0.3890951552265678, 0.38010619776354904, 0.37484145845796885, 0.36119631171317135, 0.35088304022729727, 0.34467831176169145, 0.3336804321266023, 0.32524917426451694, 0.3161362591641925, 0.30291855311957744, 0.2914916380914552]
  x_2 = [0.5678980387999357, 0.5626278738033795, 0.5559808151258164, 0.5485124822091646, 0.542564530131535, 0.5352601202474372, 0.5295085683823318, 0.5244347043472283, 0.514775700900742, 0.5051891080146059, 0.500774423515272, 0.49250598854409866, 0.4889953014293629, 0.47955528565398914, 0.4685468214804858, 0.45844062223010257, 0.4511001632226114, 0.4429911209264042, 0.43596027925587083, 0.4312430042263363]
  x_3 = [0.6483132744018151, 0.6438263699857427, 0.6382821796800353, 0.6356475138499066, 0.6282384399436303, 0.6220472698367141, 0.6197874589795659, 0.6136915687552017, 0.6072709036702543, 0.6006315740440771, 0.5926448560643518, 0.5879893678665522, 0.5821558348391975, 0.5735012503851323, 0.5711119571361145, 0.5604743225413662, 0.555664588691579, 0.5500484653577362, 0.5351790311363812, 0.526084245612221]
  x_4 = [0.7113898379507465, 0.7073249998144692, 0.7036631858644623, 0.7013173869746391, 0.6965866093454607, 0.6910111326831814, 0.6886372410690245, 0.6828887321236838, 0.678276293146115, 0.6741845170784587, 0.6691893086442214, 0.6647868515028696, 0.6596417529999575, 0.6526651590370379, 0.6473059464025841, 0.6412863604932354, 0.6354328504087017, 0.6288023385786697, 0.6228699964732717, 0.6130919851332356]
  
  x_0_25 = minus(x_0_25,x_0_25[0])
  x_0_5 = minus(x_0_5,x_0_5[0])
  x_0 = minus(x_0,x_0[0])
  x_1 = minus(x_1,x_1[0])
  x_2 = minus(x_2,x_2[0])
  x_3 = minus(x_3,x_3[0])
  x_4 = minus(x_4,x_4[0])

  import matplotlib.pyplot as plt
  plt.plot(x_0, label='x_0')
  plt.plot(x_1, label='x_1')
  plt.plot(x_2, label='x_2')
  plt.plot(x_3, label='x_3')
  plt.plot(x_4, label='x_4')
  plt.legend()
  plt.savefig("temp_gt_1.png")
  plt.close()

  plt.plot(x_0, label='x_0')
  plt.plot(x_1, label='x_1')
  plt.plot(x_0_25, label='x_0_25')
  plt.plot(x_0_5, label='x_0_5')
  plt.legend()
  plt.savefig("temp_lt_1.png")
  plt.close()

if __name__ == "__main__":
  # import numpy as np
  
  # evaluator = Evaluator(None)
  # act_func = [evaluator.emp_near] # , evaluator.act_power_0_25, evaluator.act_power_0_5, evaluator.act_power_1, evaluator.act_power_2, evaluator.act_power_3, evaluator.act_power_4
  
  # for func in act_func:
  #   res = evaluator.evaluate_debug(act_func=func)
  #   print(func, np.mean(res['average_precision']))
  #   evaluator.reset_sample_weight()

  mean_emp_far()