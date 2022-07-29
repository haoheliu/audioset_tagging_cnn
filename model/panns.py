import sys
sys.path.append('/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/audioset_tagging_cnn')
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch.models import ConvBlock, ConvBlock5x5, init_layer, init_bn
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from model.pytorch_models_mobilenet import Mobilev2Block
from model.module import TransitionBlock, BroadcastedBlock
from torchinfo import summary
from pytorch.pytorch_utils import do_mixup, interpolate, pad_framewise_output

class Audio_Frontend(nn.Module):
    """
    Wav2Mel transformation & Mel Sampling frontend
    """
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, sampler=None):
        super(Audio_Frontend, self).__init__()
  
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.sampler = sampler

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
        init_bn(self.bn0)

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)
        
        if self.sampler is not None:
            x = self.sampler(x)

        return x

class PANNS_Cnn10(nn.Module):
    def __init__(self, number_class=10, pretrained=False):
        
        super(PANNS_Cnn10, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True) 
        self.fc_audioset = nn.Linear(512, number_class, bias=True)   

        init_layer(self.fc1) 
        init_layer(self.fc_audioset) 
        
        if pretrained:
            self.load_from_ckpt()  

    def load_from_ckpt(self):
        pretrained_cnn = torch.load('pretrained_models/Cnn10.pth')['model']
        dict_new = self.state_dict().copy()
        trained_list = [i for i in pretrained_cnn.keys() if not ('fc_audioset' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
        self.load_state_dict(dict_new)


    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
      
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.2, training=self.training)
        
        clipwise_output = self.fc_audioset(x)

        output_dict = {
            'clipwise_output': clipwise_output, 
            'embedding': embedding}

        return output_dict

class PANNS_Cnn6(nn.Module):
    def __init__(self, number_class=10, pretrained=False):
        
        super(PANNS_Cnn6, self).__init__()

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True) 
        self.fc_audioset = nn.Linear(512, 10, bias=True)   

        init_layer(self.fc_audioset)  

        if pretrained: 
            self.load_from_ckpt()

    def load_from_ckpt(self):
        pretrained_cnn = torch.load('pretrained_models/Cnn6.pth')['model']
        dict_new = self.state_dict().copy()
        trained_list = [i for i in pretrained_cnn.keys() if not ('fc_audioset' in i or i.startswith('bn0') or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
        self.load_state_dict(dict_new)


    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
      
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.2, training=self.training)
        clipwise_output = self.fc_audioset(x)
        
        output_dict = {
            'clipwise_output': clipwise_output, 
            'embedding': embedding}

        return output_dict

class PANNs_MobileNet(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, number_class, pooling_factor, pooling_type):
        
        super(PANNs_MobileNet, self).__init__() 
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.conv_block1 = Mobilev2Block(in_channels=1, out_channels=16)
        self.conv_block2 = Mobilev2Block(in_channels=16, out_channels=32)
        self.conv_block3 = Mobilev2Block(in_channels=32, out_channels=64)
        self.conv_block4 = Mobilev2Block(in_channels=64, out_channels=128)
        self.conv_block5 = Mobilev2Block(in_channels=128, out_channels=128)
        self.conv_block6 = Mobilev2Block(in_channels=128, out_channels=128)

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

        self.fc1 = nn.Linear(128, 128, bias=True)
        self.fc_audioset = nn.Linear(128, number_class, bias=True)
        
        from pooling import Pooling_layer
        if(pooling_type != "no_pooling"):
            self.pooling = Pooling_layer(pooling_type, float(pooling_factor))
        else:
            self.pooling = None

        self.bn0 = nn.BatchNorm2d(64)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, x, mixup_lambda=None):
        # import ipdb; ipdb.set_trace()
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

        if(self.pooling is not None):
            x = self.pooling(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu_(self.fc1(x), negative_slope=0.01)
        embedding = F.dropout(x, p=0.2, training=self.training)
        clipwise_output = self.fc_audioset(x)
        
        output_dict = {'clipwise_output': torch.sigmoid(clipwise_output), 'embedding': embedding}

        return output_dict

class BCResNet_Mod(torch.nn.Module):
    def __init__(self, c=4, number_class=10):
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

        self.block7_1 = nn.Conv2d(int(2.5 * c), number_class, 1)
        self.block8_1 = nn.AdaptiveAvgPool2d((1, 1))

    
    def forward(self, input):
        out = input

        out = self.conv1(out)

        out = self.block1_1(out)

        out = self.block1_2(out)

        out = self.block2_1(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        
        out = self.block4_1(out)

        out = self.block5_1(out)
        out = self.block5_2(out)
        
        out = self.block6_1(out)
        out = self.block6_2(out)
        out = self.block6_3(out)
        
        out = self.block7_1(out)

        out = self.block8_1(out)
        out = self.block8_1(out)

        embedding = F.dropout(out, p=0.2, training=self.training)
        clipwise_output = torch.squeeze(torch.squeeze(out,dim=2),dim=2)

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding}

        return output_dict

if __name__ == '__main__':
    cnn6 = PANNS_Cnn6()
    mobilenet = PANNs_MobileNet()

    model = mobilenet

    k=0.025

    profile = summary(model, input_size=(1, 1, 5000, 64))
    mac1, param1 = profile.total_mult_adds, profile.total_params

    profile = summary(model, input_size=(1, 1, int(5000 * k), 64))
    mac2, param2 = profile.total_mult_adds, profile.total_params

    print(1-(mac2/mac1))



