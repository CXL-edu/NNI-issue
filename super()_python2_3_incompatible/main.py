from json import load

import torch
import nni
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import nni.retiarii.strategy as strategy
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

# from model import Net

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__.update(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update(self, *args, **kwargs):
        super(DotDict, self).update(*args, **kwargs)

def build_model(model_name: str, configs: dict):
    model_dict = {
        'test': 'from model import Net'
    }
    exec(model_dict[model_name])
    exec('print(123,{})'.format(configs))
    model = exec("Net(DotDict({})).float()".format(configs))
    # model = exec("Net(DotDict({})).float()".format(configs))
    return model


if __name__ == '__main__':
    b = load(open('setting.json'))
    b.update({'k': 3, 'y': 32, 'configs':{}})
    moedl = build_model('test', b)
    # model = Net(b)

