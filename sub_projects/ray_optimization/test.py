from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra
import os
os.environ["HYDRA_FULL_ERROR"] = "1"
@hydra.main(version_base=None, config_path="./conf", config_name="config")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    ml = instantiate(cfg)
    print(ml)


if __name__ == "__main__":
    my_app()
