from abc import ABCMeta, abstractmethod
import os
from typing import Any, Dict, Union
from matplotlib.figure import Figure

import wandb


class LoggingBackend(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.log_dict: Dict[str, Any] = {}
        self.logged_until_index = 0

    def _add_to_log(self, add_to_log: Dict[str, Any]):
        self.log_dict = {**self.log_dict, **add_to_log}

    def empty_log(self):
        self.log_dict = {}

    def log(self, log: Dict[str, Any]):
        self._add_to_log(log)
        self._log()
        self.empty_log()

    @abstractmethod 
    def log_config(self, config: Dict):
        pass

    @abstractmethod
    def _log(self):
        pass

    @abstractmethod
    def image(key: Union[str, int], figure: Figure):
        pass


class WandbLoggingBackend(LoggingBackend):
    def __init__(
        self, logging_entity: str, project_name: str, study_name: str, logging: bool
    ):
        super().__init__()
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        self.handle = wandb.init(
            entity=logging_entity,
            project=project_name,
            name=study_name,
            mode="online" if logging else "disabled",
        )

    def _log(self):
        self.handle.log(self.log_dict)
    @staticmethod
    def image(image: Figure):
        return wandb.Image(image)
    def log_config(self, config: Dict):
        self.handle.config.update(config)

