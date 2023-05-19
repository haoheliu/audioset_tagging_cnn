import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.models import init_layer
# from models.frontend import Audio_Frontend
from model.module import ConvBlock3x3, TransitionBlock, BroadcastedBlock
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from pytorch.pytorch_utils import do_mixup, interpolate, pad_framewise_output

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
class BCResNet_Mod(torch.nn.Module):
    def __init__(self, sample_rate=None, window_size=None, hop_size=None, mel_bins=None, fmin=None, fmax=None, number_class=None, pooling_factor=None, pooling_type=None,c=4, num_class=10, frontend=None, norm=False):
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        self.lamb = 0.1
        super(BCResNet_Mod, self).__init__()
        c = 10 * c
        self.conv1 = nn.Conv2d(1, 2 * c, 5, stride=(2, 2), padding=(2, 2))
        self.block1_1 = TransitionBlock(2 * c, c)
        self.block1_2 = BroadcastedBlock(c)

        self.block2_1 = nn.MaxPool2d(2)

        self.block3_1 = TransitionBlock(c, int(1.5 * c))
        self.block3_2 = BroadcastedBlock(int(1.5 * c))

        self.block4_1 = nn.MaxPool2d(2)

        self.block5_1 = TransitionBlock(int(1.5 * c), int(2 * c))
        self.block5_2 = BroadcastedBlock(int(2 * c))

        self.block6_1 = TransitionBlock(int(2 * c), int(2.5 * c))
        self.block6_2 = BroadcastedBlock(int(2.5 * c))
        self.block6_3 = BroadcastedBlock(int(2.5 * c))

        self.block7_1 = nn.Conv2d(int(2.5 * c), num_class, 1)

        self.block8_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = norm
        # self.fc_audioset = nn.Linear(1, num_class, bias=True)
        
        self.bn0 = nn.BatchNorm2d(64)
        
        if norm:
            self.one = nn.InstanceNorm2d(1)
            self.two = nn.InstanceNorm2d(int(1))
            self.three = nn.InstanceNorm2d(int(1))
            self.four = nn.InstanceNorm2d(int(1))
            self.five = nn.InstanceNorm2d(int(1))

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

    def forward(self, x, add_noise=False, training=False, noise_lambda=0.1, k=2, mixup_lambda=None):

        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
            
        if self.training:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            
        if self.frontend is not None:
            out = self.frontend(x)
        # Input: (batch_size, 1, T, F)
        else:
            out = x
            
        if self.norm:
            out = self.lamb * out + self.one(out)
        out = self.conv1(out)

        out = self.block1_1(out)

        out = self.block1_2(out)
        if self.norm:
            out = self.lamb * out + self.two(out)

        out = self.block2_1(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        if self.norm:
            out = self.lamb * out + self.three(out)

        out = self.block4_1(out)

        out = self.block5_1(out)
        out = self.block5_2(out)
        if self.norm:
            out = self.lamb * out + self.four(out)

        out = self.block6_1(out)
        out = self.block6_2(out)
        out = self.block6_3(out)
        embedding = F.dropout(out, p=0.2, training=training)
        embedding = self.block8_1(embedding)
        embedding = self.block8_1(embedding)
        if self.norm:
            out = self.lamb * out + self.five(out)
        if not training and add_noise is True:
            x_hat = []
            for i in range(k):
                feat = out
                noise = (torch.rand(feat.shape) - 0.5).to('cuda') * noise_lambda * torch.std(feat)
                feat += noise
                feat = self.block7_1(feat)

                feat = self.block8_1(feat)
                feat = self.block8_1(feat)

                clipwise_output = torch.squeeze(torch.squeeze(feat, dim=2), dim=2)
                x_hat.append(clipwise_output)
            clipwise_output = x_hat

        else:
            out = self.block7_1(out)

            out = self.block8_1(out)
            out = self.block8_1(out)

            clipwise_output = torch.squeeze(torch.squeeze(out, dim=2), dim=2)

        output_dict = {
            'clipwise_output': torch.sigmoid(clipwise_output),
            'embedding': embedding}

        return output_dict
