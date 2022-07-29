import sys
sys.path.append('/vol/research/MachineAudition_CVSSP/xl01061/audioset_tagging_cnn/')
sys.path.append('/vol/research/MachineAudition_CVSSP/xl01061/audioset_tagging_cnn/network')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from pytorch.pytorch_utils import do_mixup, interpolate, pad_framewise_output
from init_weight import init_layer, init_bn
from pytorch.models import ConvBlock, ConvBlock5x5
from others.netvlad.VLAD import NetVLAD, SeqVLAD, DeTVLAD, SpecVLAD
from pooling import Pooling_layer
import time
from thop import profile
from thop import clever_format

class Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, frontend=None):
        
        super(Cnn6, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.frontend = frontend

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # x = torch.randn(1, 1, 1501, 64)
        mel = x
        
        if self.training:
            x = self.spec_augmenter(x)
        
        if self.frontend is not None:
            x = self.frontend(x)

        vlad = x

        # print(x.shape)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        tf_embed = x
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        last_embed = x
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {
            'clipwise_output': clipwise_output, 
            'embedding': embedding, 
            'TF-Embed': tf_embed, 
            'last_embed': last_embed,
            'vlad': vlad,
            'mel': mel}

        return output_dict

if __name__ == '__main__':
    model_params = {'sample_rate': 48000,
                    'window_size': 1024,
                    'hop_size': 320,
                    'mel_bins': 64,
                    'fmin': 50,
                    'fmax': 14000,
                    'classes_num': 10}

    vlad1 = Pooling_layer(pooling_type='max', factor=32)
    model1 = Cnn6(frontend=vlad1, **model_params)
    model1.to('cpu')

    # vlad2 = SeqVLAD(cluster_size=64, feature_size=64, bidirectional=True)
    # model2 = Cnn6(frontend=vlad2, **model_params)
    # model2.to('cpu')

    # vlad3 = SpecVLAD(window_len=44, hop_len=44, cluster_size=32, feature_size=64)
    # model3 = Cnn6(frontend=vlad3, **model_params)
    # model3.to('cpu')

    
    model4 = Cnn6(**model_params)
    model4.to('cpu')

    model1.eval()
    # model2.eval()
    # model3.eval()
    model4.eval()


    spec = Spectrogram(n_fft=1024, hop_length=320, 
            win_length=1024, window='hann', center=True, pad_mode='reflect', 
            freeze_parameters=True)

    input = torch.randn(1, 480000).to('cpu')

    macs, params = profile(spec, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print('spec', macs, params)

    macs, params = profile(model4, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print('model4', macs, params)

    # macs, params = profile(model3, inputs=(input, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('model3', macs, params)

    # macs, params = profile(model2, inputs=(input, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('model2', macs, params)

    macs, params = profile(model1, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print('model1', macs, params)

    

    # with torch.no_grad():
    #     time1 = time.time()
    #     output1 = model1(input)
    #     time2 = time.time()
    #     output2 = model2(input)
    #     time3 = time.time()
    #     output3 = model3(input)
    #     time4 = time.time()

    #     print(time2-time1)
    #     print(time3-time2)
    #     print(time4-time3)
    # pass
    