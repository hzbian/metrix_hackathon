from abc import ABCMeta, abstractmethod
import os
from typing import Any
from matplotlib.figure import Figure
import plotly.graph_objects as go
import wandb


class LoggingBackend(metaclass=ABCMeta):
    @abstractmethod
    def log(self, log: dict[str, Any]):
        pass

    @abstractmethod 
    def log_config(self, config: dict):
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

    def log(self, log: dict):
        assert self.handle is not None
        self.handle.log(log)
    def figure_to_image(self, figure: Figure | go.Figure):
        if isinstance(figure, Figure):
            return wandb.Image(figure)
        else:
            return figure
    def log_config(self, config: dict):
        assert self.handle is not None
        self.handle.config.update(config)

class DebugPlotBackend(LoggingBackend):
    def __init__(self, path: str = 'outputs/'):
        super().__init__()
        self.path = path
    def figure_to_image(self, figure: Figure):
        return figure
    def log(self, log: dict):
        for key, value in log.items():
            if isinstance(value, Figure):
                value.savefig(self.path+str(key)+".png")
            if isinstance(value, go.Figure):
                value.write_html(self.path+str(key)+".html")
        pass
    def log_config(self, _: dict):
        pass