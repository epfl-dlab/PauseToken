try:
    import wandb
except ImportError:
    raise ImportError("Please install wandb using pip install wandb")

from stable_baselines3.common.logger import Logger
from typing import Any, Optional, Tuple, Union, List
import os
import tempfile
import datetime
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.logger import make_output_format

class WandbLogger(Logger):
    def __init__(self, folder: Optional[str], output_formats: List[KVWriter], project: str, name: str, config: dict = {}, notes: str = None):
        if folder is None:
            folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S-%f"))
        
        assert isinstance(folder, str)
        os.makedirs(folder, exist_ok=True)
        
        if output_formats is None:
            log_suffix = ""
            format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")            
            format_strings = list(filter(None, format_strings))
            output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]
        
        super().__init__(folder, output_formats)
        wandb.init(project=project, name=name, config=config, notes=notes)
    
    def record(self, key: str, value: Any, exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        super().record(key, value, exclude)
        wandb.log({key: self.name_to_value[key]})
    
    def log_hyperparams(self,hparams):
        wandb.config.update(hparams)
        