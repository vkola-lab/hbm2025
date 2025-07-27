import os
import sys
import numpy as np
import itertools
from collections import Counter
import pandas as pd

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb
import torch

# torch.autograd.set_detect_anomaly(True)
from torch.distributed import init_process_group, destroy_process_group
import os
from config import MyConfig
from trainer import Trainer
from datetime import timedelta
from utils.dist_utils import init_distributed_mode
from utils.visualization_utils import write_scores, read_raw_score, plot_curve_multiple
from utils.stat_utils import get_metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


def ddp_setup():
    print(f"-------------- Setting up ddp {os.environ['LOCAL_RANK']}...")
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def _main(cfg: MyConfig):
    use_ddp = (cfg.hardware.multi_gpus == "ddp") and torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.autograd.set_detect_anomaly(False)

        init_distributed_mode(cfg.hardware)

    num_gpus = torch.cuda.device_count()
    ## add num_gpus in hardward config
    if "num_gpus" not in cfg["hardware"]:
        OmegaConf.update(cfg, "hardware.num_gpus", num_gpus, force_add=True)
    
    # print(OmegaConf.to_yaml(cfg))
    # print(type(cfg.modal_prompt.embed_module), cfg.modal_prompt.embed_module)
    # exit()        

    trainer = Trainer(cfg)
    
    if cfg.test:
        if cfg.dataset.name.lower() in ['brats2018', 'brats2020', 'gliomapost']:
            trainer.validate_SEG_BRATS(trainer.test_loader)
            exit(0)
        else:
            raise NotImplementedError

    else:
        print("starting trainer.train()...")
        trainer.train()

    if use_ddp:
        destroy_process_group()


@hydra.main(version_base=None, config_path="configs", config_name="main_cfg")
def main(cfg: MyConfig) -> None:
    if cfg.logging.wandb.use_wandb and cfg.logging.wandb.sweep_id:
        wandb.agent(cfg.logging.wandb.sweep_id, function=lambda: _main(cfg), count=1)
    else:
        _main(cfg)

if __name__ == "__main__":
    main()