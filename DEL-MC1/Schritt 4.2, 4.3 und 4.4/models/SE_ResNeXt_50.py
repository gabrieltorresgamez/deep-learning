# https://lightning.ai/docs/pytorch/latest/advanced/transfer_learning.html

import torch
import torch.nn as nn
import torchvision.models as models

from ._IModel import IModel


class ResNeXt50(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)

        # init a pretrained resnet
        backbone = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1, progress=True
        )
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x
