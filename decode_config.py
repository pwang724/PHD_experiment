import CONSTANTS.conditions as conditions
from CONSTANTS.constants import constants
import os

class BaseConfig(object):
    def __init__(self):
        pass

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

class DecodeConfig(BaseConfig):
    def __init__(self):
        super(DecodeConfig, self).__init__()
        self.condition = conditions.OFC
        self.decode_style = 'valence'
        self.neurons = 20
        self.shuffle = False

        self.save_path = os.path.join(constants.LOCAL_ANALYSIS_PATH, self.condition.name, self.decode_style)
        self.data_path = os.path.join(constants.LOCAL_DATA_PATH, constants.LOCAL_DATA_TIMEPOINT_FOLDER,
                                 self.condition.name)