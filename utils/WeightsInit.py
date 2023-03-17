import math

import torch
import torch.nn as nn


class WeightsInit:
    def __init__(self, weights_init_type="default"):
        if weights_init_type == "kaiming_uniform":
            self.init_weights = self.__weights_init_kaiming_uniform
        elif weights_init_type == "kaiming_normal":
            self.init_weights = self.__weights_init_kaiming_normal
        elif weights_init_type == "xavier_uniform":
            self.init_weights = self.__weights_init_xavier_uniform
        elif weights_init_type == "xavier_normal":
            self.init_weights = self.__weights_init_xavier_normal
        elif weights_init_type == "default":
            self.init_weights = self.__weights_init_default
        else:
            raise NotImplementedError

    def __weights_init_kaiming_uniform(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        return module

    def __weights_init_kaiming_normal(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        return module

    def __weights_init_xavier_uniform(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        return module

    def __weights_init_xavier_normal(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        return module

    def __weights_init_default(self, module):
        return module  # Do nothing
