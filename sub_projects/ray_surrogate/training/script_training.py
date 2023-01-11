import os, sys, shutil

sys.path.insert(0, '../../../')

import wandb

import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from sub_projects.ray_surrogate.nn_models import SurrogateModel

import cfg_training as CFG

# --- Save all config files ---
cfg_files = [CFG.__file__]
os.makedirs(CFG.RUN_PATH, exist_ok=True)
for file in cfg_files:
    try:
        shutil.copy(file, os.path.join(CFG.RUN_PATH, os.path.basename(file)))
        print('Config file ' + file + ' saved to results')
    except shutil.SameFileError:
        print('Config file ' + file + ' already exists')

# --- wandb init ---
wandb.login()
wandb.finish()
wandb.init(entity='hzb-aos',
           project='metrix_hackathon_surrogate',
           name=CFG.RUN_ID,
           id=CFG.RUN_ID,
           mode='online' if CFG.WANDB_ONLINE else 'disabled',
           resume='must' if CFG.RESUME_RUN else None,
           dir=os.path.join(CFG.RUN_PATH))
wandb_logger = WandbLogger(log_model=False)

# --- PL seed ---
pytorch_lightning.seed_everything(CFG.TRAINING_SEED)

# --- Create the EmbeddingPipeline to be trained ---
if CFG.RESUME_RUN:
    inversion_net = SurrogateModel.load_from_checkpoint(os.path.join(CFG.RUN_PATH, 'last.ckpt'))
else:
    inversion_net = SurrogateModel(base_net=CFG.BASE_NET[0],
                                   base_net_params=CFG.BASE_NET[1],
                                   param_preprocessor=CFG.PARAM_PREPROCESSOR[0],
                                   param_preprocessor_params=CFG.PARAM_PREPROCESSOR[1],
                                   hist_subsampler=CFG.HIST_SUBSAMPLER[0],
                                   hist_subsampler_params=CFG.HIST_SUBSAMPLER[1],
                                   hist_to_pc=CFG.HIST_TO_PC[0],
                                   hist_to_pc_params=CFG.HIST_TO_PC[1],
                                   param_dim=CFG.PARAM_DIM,
                                   pc_supp_dim=CFG.PC_SUPP_DIM,
                                   loss_func=CFG.LOSS_FUNC[0],
                                   loss_func_params=CFG.LOSS_FUNC[1],
                                   optimizer=CFG.OPTIMIZER[0],
                                   optimizer_params=CFG.OPTIMIZER[1],
                                   scheduler=CFG.SCHEDULER[0],
                                   scheduler_params=CFG.SCHEDULER[1],
                                   val_metrics=CFG.VAL_METRICS)

# --- PL training ---
checkpoint_callback = ModelCheckpoint(dirpath=CFG.RUN_PATH,
                                      filename='best_val',
                                      monitor=CFG.MONITOR_VAL_LOSS,
                                      mode='min',
                                      save_last=True)

trainer = Trainer(default_root_dir=CFG.RUN_PATH,
                  logger=wandb_logger,
                  log_every_n_steps=50,
                  num_sanity_val_steps=CFG.NUM_SANITY_VAL_STEPS,
                  callbacks=[checkpoint_callback] + CFG.CALLBACKS,
                  limit_train_batches=CFG.FRAC_TRAIN_SAMPLES,
                  limit_val_batches=CFG.FRAC_VAL_SAMPLES,
                  max_epochs=CFG.MAX_EPOCHS,
                  accelerator=CFG.DEVICE,
                  devices=[CFG.GPU_ID])

trainer.fit(model=inversion_net,
            train_dataloaders=CFG.TRAIN_DATALOADER,
            val_dataloaders=CFG.VAL_DATALOADER,
            ckpt_path=os.path.join(CFG.RUN_PATH, 'last.ckpt') if CFG.RESUME_RUN else None)

# --- Wrap up ---
wandb.finish()
