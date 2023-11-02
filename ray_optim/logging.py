from abc import ABCMeta, abstractmethod
import os
from typing import Any, Dict, Union
from matplotlib.figure import Figure
import plotly.graph_objects as go
import wandb


class LoggingBackend(metaclass=ABCMeta):
    @abstractmethod
    def log(self, log: Dict[str, Any]):
        pass

    @abstractmethod 
    def log_config(self, config: Dict):
        pass

    @abstractmethod
    def figure_to_image(self, figure: Figure):
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

    def log(self, log: Dict):
        self.handle.log(log)
    def figure_to_image(self, figure: Figure):
        return wandb.Image(figure)
    def log_config(self, config: Dict):
        self.handle.config.update(config)

class DebugPlotBackend(LoggingBackend):
    def __init__(self, path: str = 'outputs/'):
        super().__init__()
        self.path = path
    def figure_to_image(self, figure: Figure):
        return figure
    def log(self, log: Dict):
        for key, value in log.items():
            if isinstance(value, Figure):
                value.savefig(self.path+str(key)+".png")
            if isinstance(value, go.Figure):
                value.write_html(self.path+str(key)+".html")
        pass
    def log_config(self, _: Dict):
        pass