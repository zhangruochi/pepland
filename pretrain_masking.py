import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import nni
import hydra
import mlflow
import shutil
import timeit
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from model.model import PharmHGT 
from model.hgt import HGT,HeteroRGCN
from model.data import make_loaders, MaskAtom
from utils.distribution import setup_multinodes, cleanup_multinodes
from utils.utils import fix_random_seed, get_device, is_parallel, load_weights, load_model_masking
from utils.std_logger import Logger
from trainer import Masking_Trainer



@hydra.main(config_path="configs", config_name="pretrain_masking.yaml", version_base='1.2')

# @hydra.main(config_path="configs", config_name="test.yaml", version_base='1.2')
def main(cfg: DictConfig):
    
    global_rank = 0
    local_rank = 0
    world_size = 0
    
    if not cfg.mode.nni and cfg.logger.log:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        
    # setup distuibution data parallel
    if cfg.mode.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # os.environ['NCCL_DEBUG'] = 'INFO'
        # os.environ['NCCL_SHM_DISABLE'] = '1'
        # os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
        random_seed = cfg.train.random_seed + local_rank
        setup_multinodes(local_rank, world_size)
        device = torch.device("cuda", local_rank)
    else:
        random_seed = cfg.train.random_seed
        device = get_device(cfg)

    if global_rank == 0:
        print("setting random seed: {}".format(random_seed))

    fix_random_seed(random_seed, cuda_deterministic=True)

    if global_rank == 0:
        print("num layer: {} mask rate: {:.2f} mask edge: {:.2f}".format(cfg.train.num_layer, cfg.train.mask_rate, cfg.train.mask_edge))
    # set up dataset and transform function.
    dataloaders = make_loaders( cfg=cfg,
                                ddp=cfg.mode.ddp,
                                dataset=cfg.train.dataset,
                                world_size=world_size,
                                global_rank=global_rank,
                                batch_size=cfg.train.batch_size,
                                num_workers=cfg.train.num_workers,
                                transform=MaskAtom(num_atom_type=119, 
                                                    num_edge_type=5,
                                                    mask_rate=cfg.train.mask_rate, 
                                                    mask_edge=cfg.train.mask_edge, 
                                                    mask_fragment=cfg.train.mask_pharm,
                                                    mask_amino=cfg.train.mask_amino,
                                                    mask_pep=cfg.train.mask_pep)
                                                    )   
     
     
    
    # set up models, one for pre-training and one for context embedding
    
    if cfg.train.model == 'PharmHGT':
        model = PharmHGT(cfg.train.hid_dim,cfg.train.act,cfg.train.num_layer, cfg.train.atom_dim, cfg.train.bond_dim, cfg.train.pharm_dim, cfg.train.reac_dim).to(device)
    elif cfg.train.model == 'hgt':
        node_dict = {'a':0,'p':1}
        edge_dict = {('a','b','a'):0,('p','r','p'):1,('a','j','p'):2,('p','j','a'):2}
        model = HGT(node_dict,
                    edge_dict,
                    cfg.train.atom_dim,
                    cfg.train.pharm_dim,
                    cfg.train.hid_dim,
                    n_layers = cfg.train.num_layer,
                    n_heads = 4,
                    use_norm = True).to(device)
    elif cfg.train.model == 'fine-tune':
        model,_,_ = load_model_masking(cfg.inference.model_path,device)
    linear_pred_atoms = torch.nn.Linear(cfg.train.hid_dim, 119).to(device)
    linear_pred_pharms = torch.nn.Linear(cfg.train.hid_dim, 264).to(device)
    # linear_pred_amino = torch.nn.Linear(cfg.train.hid_dim, 21).to(device)
    linear_pred_bonds = torch.nn.Linear(cfg.train.hid_dim, 4).to(device)

    # model_list = [model, linear_pred_atoms, linear_pred_pharms, linear_pred_amino, linear_pred_bonds]
    model_list = [model, linear_pred_atoms, linear_pred_pharms, linear_pred_bonds]

    if cfg.mode.ddp:
        model_list = [DDP(model, device_ids=[global_rank], output_device=global_rank) for model in model_list]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    optimizer_linear_pred_pharms = optim.Adam(linear_pred_pharms.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    
    # optimizer_linear_pred_amino = optim.Adam(linear_pred_amino.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.decay)
    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_pharms, optimizer_linear_pred_bonds]    
    # optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_pharms, optimizer_linear_pred_amino, optimizer_linear_pred_bonds]    
    criterion = nn.CrossEntropyLoss()

    if not cfg.mode.nni and cfg.logger.log and global_rank == 0:
        # log hyper-parameters
        for p, v in cfg.train.items():
            mlflow.log_param(p, v)
    
    trainer = Masking_Trainer(
                            cfg=cfg, 
                            global_rank=global_rank, 
                            world_size=world_size, 
                            model_list=model_list, 
                            dataloaders=dataloaders, 
                            criterion=criterion, 
                            optimizer_list=optimizer_list, 
                            device=device,
                            output_dir=cfg.logger.log_dir)
    trainer.run()
    

    if cfg.logger.log:
        if global_rank == 0:
            Logger.info("finished training......")
    #         Logger.info("start evaluating......")
    #         Logger.info("loading best weights from {}......".format(trainer.best_model_path))
            
    # model = load_weights(model, os.path.join(cfg.logger.log_dir, trainer.best_model_path, 'model'), device)
    # linear_pred_atoms = load_weights(linear_pred_atoms, os.path.join(cfg.logger.log_dir, trainer.best_model_path, 'linear_pred_atoms'), device)
    # linear_pred_bonds = load_weights(linear_pred_bonds, os.path.join(cfg.logger.log_dir, trainer.best_model_path, 'linear_pred_bonds'), device)
    
    # # final evaluate
    # split = 'test'
    # evaluator = Masking_Evaluator(model, linear_pred_atoms, linear_pred_bonds, 
    #                               dataloaders[split], criterion, device, cfg)
    # test_metrics = evaluator.run()
    
    # if global_rank == 0:
    #     for metric_name, metric_v in test_metrics.items():
    #         if isinstance(metric_v,  (float, np.float, int, np.int)):
    #             metric_v = round(metric_v,5)
    #         elif isinstance(metric_v,  str):
    #             metric_v = "\n" + metric_v
    #         Logger.info("{} | {}: {}".format(split, metric_name, metric_v))
            
    #     if cfg.mode.nni:
    #         # report final result
    #         test_metrics["default"] = test_metrics["test_{}".format(cfg.task[cfg.task.type].default_metric)]
    #         nni.report_final_result(test_metrics)

    #     if not cfg.mode.nni and cfg.logger.log:            
    #         for metric_name, metric_v in test_metrics.items():
    #             if isinstance(metric_v, (float, np.float64, int, np.int32, np.int64)):
    #                 mlflow.log_metric("test_final/{}".format(metric_name), metric_v, step=1)
    #             elif isinstance(metric_v, str):
    #                 mlflow.log_text(metric_v, "test_final/report.txt")
    
    if cfg.mode.ddp:
        cleanup_multinodes()
    
if __name__ == "__main__":
    main()
