import torch
import torch.nn as nn
import torch.nn.functional as F
from specpool2d import SpectralPool2d

class Pooling_layer(nn.Module):
    """
    pooling_type: 'avg', 'max', 'avg-max', 'uniform', 'spec'
    factor: temporal pooling factor in (0, 1)
    """
    def __init__(self, pooling_type='avg-max', factor=0.75):
        super(Pooling_layer, self).__init__()
        self.factor = factor
        self.pooling_type = pooling_type
        
        if self.pooling_type == 'spec':
            self.SpecPool2d = SpectralPool2d(scale_factor=(factor, 1))

    def forward(self, x):
        """
        args:
            x: input mel spectrogram [batch, 1, time, frequency] 
        return:
            out: reduced features [batch, 1, time * factor, frequency]
        """
        factor = int(x.shape[2] * self.factor)

        if self.pooling_type == 'avg':
            size = x.shape[2] // factor
            out = F.avg_pool2d(x, kernel_size=(size, 1))
        elif self.pooling_type == 'max':
            size = x.shape[2] // factor
            out =  F.max_pool2d(x, kernel_size=(size, 1))
        elif self.pooling_type == 'avg-max':
            size = x.shape[2] // factor
            out1 =  F.max_pool2d(x, kernel_size=(size, 1))
            out2 = F.avg_pool2d(x, kernel_size=(size, 1))
            out = out1 + out2
        elif self.pooling_type == 'uniform':
            out =  self.uniform_sample(x, factor)
        elif self.pooling_type == 'spec':
            out = self.SpecPool2d(x)

        
        return out

    def uniform_sample(self, input, factor):
        """
            args:
                x: input mel spectrogram [batch, 1, time, frequency] 
            return:
                out: reduced features [batch, 1, factor, frequency]
            """
        indexes = torch.linspace(0, input.shape[2]-1, factor).tolist()
        indexes = [int(num) for num in indexes]
        
        output = input[:, :, indexes, :]

        return output

