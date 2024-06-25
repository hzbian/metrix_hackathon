from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_nn.nn.xy_hist_data_models import MetrixXYHistSurrogate
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
import glob
from ray_nn.data.transform import Select
from ray_nn.nn.xy_hist_data_models import StandardizeXYHist
from ray_tools.simulation.torch_datasets import (
    BalancedMemoryDataset,
    MemoryDataset,
    RayDataset,
)
import glob
from datasets.metrix_simulation.config_ray_emergency_surrogate import (
    PARAM_CONTAINER_FUNC as params,
)
from lightning.pytorch.loggers import WandbLogger

load_len: int | None = None
dataset_normalize_outputs = True
h5_files = list(
    glob.iglob("datasets/metrix_simulation/ray_emergency_surrogate/50+50_data_raw_*.h5")
)  # ['datasets/metrix_simulation/ray_emergency_surrogate/49+50_data_raw_0.h5']
dataset = RayDataset(
    h5_files=h5_files,
    sub_groups=["1e5/params", "1e5/histogram", "1e5/n_rays"],
    transform=Select(
        keys=["1e5/params", "1e5/histogram", "1e5/n_rays"],
        search_space=params(),
        non_dict_transform={"1e5/histogram": StandardizeXYHist()},
    ),
)

memory_dataset = BalancedMemoryDataset(
    dataset=dataset, load_len=load_len, min_n_rays=499
)
datamodule = DefaultDataModule(dataset=memory_dataset, num_workers=3)
datamodule.prepare_data()
model = MetrixXYHistSurrogate(
    dataset_length=load_len, dataset_normalize_outputs=dataset_normalize_outputs
)
wandb_logger = WandbLogger(
    name="ref_bal_499_sch_.999_test", project="xy_hist", save_dir="outputs"
)
# wandb_logger = None
datamodule.setup(stage="test")

lr_monitor = LearningRateMonitor(logging_interval="step")
trainer = L.Trainer(
    max_epochs=9999,
    logger=wandb_logger,
    log_every_n_steps=100,
    check_val_every_n_epoch=1,
    callbacks=[lr_monitor],
)
trainer.init_module()

trainer.test(
    datamodule=datamodule,
    ckpt_path="outputs/xy_hist/pd387nv8/checkpoints/epoch=755-step=147239316.ckpt",
    model=model,
)
