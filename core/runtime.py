# runtime.py
from .config import CONFIG
from . import constants
from data.data_manager import DataManager
from .paths import Paths

class RUNTIME:
    def __init__(self):
        self._consts = constants
        self._cfg = None
        self._paths = None
        self._data_manager = None

        self._monkey_name = None
        self._z_score_index = None
        self._analysis_type = None
        self._group_size = None

    @property
    def cfg(self):
        return self._cfg
    
    @property
    def consts(self):
        return self._consts
    
    @property
    def data_manager(self):
        return self._data_manager
    
    @property
    def paths(self):
        return self._paths

    @property
    def monkey_name(self):
        return self._monkey_name
    
    @property
    def z_score_index(self):
        return self._z_score_index
    
    @property
    def analysis_type(self):
        return self._analysis_type
    
    @property
    def group_size(self):
        return self._group_size

    @cfg.setter
    def cfg(self, value):
        self._cfg = value
    
    @paths.setter
    def paths(self, value):
        self._paths = value
    
    @data_manager.setter
    def data_manager(self, value):
        self._data_manager = value

    @monkey_name.setter
    def monkey_name(self, value):
        self._monkey_name = value
    
    @z_score_index.setter
    def z_score_index(self, value):
        self._z_score_index = value
    
    @analysis_type.setter
    def analysis_type(self, value):
        self._analysis_type = value
    
    @group_size.setter
    def group_size(self, value):
        self._group_size = value
    
    
    def update(self, monkey_name: str, z_score_index: int, analysis_type: str = None, group_size: int = None):
        self._monkey_name = monkey_name
        self._z_score_index = z_score_index
        self._analysis_type = analysis_type
        self._group_size = group_size

        self._cfg = CONFIG(monkey_name, z_score_index)
        self._paths = Paths(self._cfg, self._consts)
        self._data_manager = DataManager(self._cfg)


runtime = RUNTIME()