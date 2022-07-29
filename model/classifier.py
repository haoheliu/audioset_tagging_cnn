import torch
import torch.nn as nn
from model.panns import Audio_Frontend, PANNS_Cnn10
from model.pooling import Pooling_layer

class Classifier(nn.Module):
    def __init__(self, frontend, backbone):
        super(Classifier, self).__init__()
        self.frontend = frontend
        self.backbone = backbone

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """
        return self.backbone(self.frontend(input))

if __name__ == '__main__':
    panns_params = {
        'sample_rate': 48000, 
        'window_size': 1024, 
        'hop_size': 320, 
        'mel_bins': 64, 
        'fmin': 50, 
        'fmax': 14000}

    sampler = Pooling_layer(pooling_type='max', factor=64)
    frontend = Audio_Frontend(**panns_params, sampler=sampler)
    backbone = PANNS_Cnn10(number_class=10, pretrained=True)

    model = Classifier(frontend=frontend, backbone=backbone)

    print(model(torch.randn(8, 480000))['clipwise_output'])