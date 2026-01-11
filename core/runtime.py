# runtime.py
from .config import CONFIG
from . import constants

class RUNTIME:
    def __init__(self):
        self._cfg = None
        self._consts = constants

    def get_cfg(self):
        return self._cfg
    
    def get_consts(self):
        return self._consts
    
    def get_data_manager(self):
        return self._data_manager

    def set_cfg(self, monkey_name: str, z_score_index: int):
        self._cfg = CONFIG(monkey_name, z_score_index)
        from data.data_manager import DataManager
        self._data_manager = DataManager(self._cfg)

runtime = RUNTIME()