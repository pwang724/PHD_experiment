import _CONSTANTS.conditions as conditions
from _CONSTANTS.config import Config
import os

class BaseConfig(object):
    def __init__(self):
        pass

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

class DecodeConfig(BaseConfig):
    def __init__(self):
        super(DecodeConfig, self).__init__()
        self.decode_style = 'valence'
        self.neurons = 20
        self.repeat = 2
        self.shuffle = False
