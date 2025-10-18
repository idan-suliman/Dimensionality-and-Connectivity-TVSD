from config import CONFIG
import constants

class RUNTIME:
    def __init__(self):
        self._cfg = None
        self._consts = constants

    def get_cfg(self):
        return self._cfg
    
    def get_consts(self):
        return self._consts
    
    def set_cfg(self, monkey_name: str, z_score_index: int):
        self._cfg = CONFIG(monkey_name, z_score_index)

runtime = RUNTIME()