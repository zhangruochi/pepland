import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nni
import torch
import mlflow
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path

from model.util import compute_accuracy
from utils.std_logger import Logger
from utils.utils import is_parallel

# from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.distributed import ReduceOp


class Masking_Trainer(object):
    def __init__(self, 
                 cfg, 
                 global_rank, 
                 world_size, 
                 model_list, 
                 dataloaders, 
                 criterion, 
                 optimizer_list, 
                 device, 
                 output_dir):
        self.cfg = cfg
        self.global_rank = global_rank
        self.world_size = world_size
        # self.model, self.linear_pred_atoms, self.linear_pred_pharms, self.linear_pred_amino, self.linear_pred_bonds = model_list
        
        self.model, self.linear_pred_atoms, self.linear_pred_pharms, self.linear_pred_bonds = model_list
        self.optimizer_model, self.optimizer_linear_pred_atoms, self.optimizer_linear_pred_pharms, self.optimizer_linear_pred_bonds = optimizer_list
        # self.optimizer_model, self.optimizer_linear_pred_atoms, self.optimizer_linear_pred_pharms, self.optimizer_linear_pred_amino, self.optimizer_linear_pred_bonds = optimizer_list
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.device = device
        self.epoch = 0
        
        self.global_train_step = 0
        self.global_valid_eval_step = 0
        self.global_train_eval_step = 0
        self.global_test_eval_step = 0
        
        self.default_metric = 'acc_atom'
        self.best_metric = -1
        self.best_model_path = Path(".")

        self.root_level_dir = os.getcwd()
        self.output_dir = output_dir
        
    def train_epoch(self):
        
        self.model.train()
        self.linear_pred_atoms.train()
        self.linear_pred_pharms.train()
        # self.linear_pred_amino.train()
        self.linear_pred_bonds.train()

        loss_accum = 0
        acc_atom_accum = 0
        acc_pharm_accum = 0
        # acc_amino_accum = 0
        acc_edge_accum = 0
        
        if self.cfg.mode.ddp:
            self.dataloaders['train'].sampler.set_epoch(self.epoch)
            
        for step, batch in enumerate(self.dataloaders['train']):
            
            batch = batch.to(self.device)
            # print(batch.nodes['a'].data['f'][batch.nodes['a'].data['mask']==True])
            # assert False
            atom_rep,pharm_rep = self.model(batch)

            ## loss for atoms
            pred_atom = self.linear_pred_atoms(atom_rep[batch.nodes['a'].data['mask']==True])
            # print(pred_atom.double().shape)
            # print(batch.nodes['a'].data['label'][batch.nodes['a'].data['mask']==True][:,0].shape)
            loss = self.criterion(pred_atom.double(), batch.nodes['a'].data['label'][batch.nodes['a'].data['mask']==True][:,0])
            acc_atom = compute_accuracy(pred_atom, batch.nodes['a'].data['label'][batch.nodes['a'].data['mask']==True][:,0])
            acc_atom_accum += acc_atom
            
            # if self.cfg.train.mask_amino:
            #     pred_amino = self.linear_pred_amino(atom_rep[batch.nodes['a'].data['mask']==True])
            #     loss += self.criterion(pred_amino.double(), batch.nodes['a'].data['aa_label'][batch.nodes['a'].data['mask']==True])
            #     acc_amino = compute_accuracy(pred_amino, batch.nodes['a'].data['aa_label'][batch.nodes['a'].data['mask']==True])
            #     acc_amino_accum += acc_amino
                
            if self.cfg.train.mask_pharm:
                pred_pharm = self.linear_pred_pharms(pharm_rep[batch.nodes['p'].data['mask']==True])
                loss += self.criterion(pred_pharm.double(), batch.nodes['p'].data['label'][batch.nodes['p'].data['mask']==True])
                acc_pharm = compute_accuracy(pred_pharm, batch.nodes['p'].data['label'][batch.nodes['p'].data['mask']==True])
                acc_pharm_accum += acc_pharm
                    
            if self.cfg.train.mask_edge:
                masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
                edge_rep = atom_rep[masked_edge_index[0]] + atom_rep[masked_edge_index[1]]
                pred_edge = self.linear_pred_bonds(edge_rep)
                loss += self.criterion(pred_edge.double(), batch.mask_edge_label[:,0])

                acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])
                acc_edge_accum += acc_edge

            self.optimizer_model.zero_grad()
            self.optimizer_linear_pred_atoms.zero_grad()
            self.optimizer_linear_pred_pharms.zero_grad()
            # self.optimizer_linear_pred_amino.zero_grad()
            self.optimizer_linear_pred_bonds.zero_grad()

            loss.backward()

            self.optimizer_model.step()
            self.optimizer_linear_pred_atoms.step()
            self.optimizer_linear_pred_pharms.step()
            # self.optimizer_linear_pred_amino.step()
            self.optimizer_linear_pred_bonds.step()

            loss_accum += float(loss.cpu().item())
            
            if (self.global_train_step+1) % self.cfg.logger.log_per_steps == 0:
                Logger.info("train | epoch: {:d}  step: {:d} | loss: {:.4f}".format(
                    self.epoch, self.global_train_step, loss_accum/(step+1)))
                
                if not self.cfg.mode.nni and self.global_rank == 0 and self.cfg.logger.log:
                    mlflow.log_metric("train/loss", loss_accum/(step+1), step=self.global_train_step)
                    mlflow.log_metric("train/acc_atom", acc_atom_accum/(step+1), step=self.global_train_step)
                    mlflow.log_metric("train/acc_pharm", acc_pharm_accum/(step+1), step=self.global_train_step)
                    # mlflow.log_metric("train/acc_amino", acc_amino_accum/(step+1), step=self.global_train_step)
                    mlflow.log_metric("train/acc_edge", acc_edge_accum/(step+1), step=self.global_train_step)
            if (self.global_train_step+1) % 500==0:
                self.eval_epoch("valid")    
                self.eval_epoch("test")

            self.global_train_step += 1

    def eval_epoch(self, split):

        metrics = self.evaluate(split)

        if split == "test":
            self.global_test_eval_step += 1
            step = self.global_test_eval_step
        elif split == "valid":
            self.global_valid_eval_step += 1
            step = self.global_valid_eval_step
        elif split == "train":
            self.global_train_eval_step += 1
            step = self.global_test_eval_step

        if self.global_rank == 0:

            for metric_name, metric_v in metrics.items():
                if isinstance(metric_v,  (float, np.float, int, np.int)):
                    metric_v = round(metric_v,5)
                elif isinstance(metric_v,  str):
                    metric_v = "\n" + metric_v
                Logger.info("{} |  step: {} | {}: {}".format(split, step, metric_name, metric_v))
                    

            if not self.cfg.mode.nni and self.global_rank == 0 and self.cfg.logger.log:
                for metric_name, metric_v in metrics.items():
                    if isinstance(metric_v,  (float, np.float, int, np.int)):
                        mlflow.log_metric("{}/{}".format(split, metric_name),
                                        metric_v,
                                        step=step)
                    elif isinstance(metric_v, str):
                        mlflow.log_text(metric_v, "{}/{}_report.txt".format(split, step))

        if split == "valid":

            # learning rate scheduler
            # if self.cfg.train.lr_scheduler.type in ("plateau",):
            #     self.scheduler.step(metrics["{}_{}".format(split, self.default_metric)])
            
            # report intermediate result for nni
            if self.cfg.mode.nni:
                metrics["default"] = metrics["{}_{}".format(split, self.default_metric)]
                nni.report_intermediate_result(metrics)

            if metrics["{}_{}".format(split, self.default_metric)] >= self.best_metric:
                self.best_metric = metrics["{}_{}".format(split, self.default_metric)]

                self.best_model_path = Path("model_step_{}_{}_{}".format(
                    self.global_valid_eval_step, self.default_metric, round(metrics["{}_{}".format(split, self.default_metric)], 3)))

                if self.global_rank == 0 and self.best_model_path.exists():
                    shutil.rmtree(self.best_model_path)

                if self.global_rank == 0 and self.cfg.logger.log:
                    mlflow.pytorch.save_model(
                        pytorch_model=(self.model.module if is_parallel(self.model) else self.model),
                        path=os.path.join(self.output_dir, self.best_model_path, 'model'),
                        code_paths=[os.path.join(self.root_level_dir, "model"),
                                    os.path.join(self.root_level_dir, "utils")])
                    mlflow.pytorch.save_model(
                        pytorch_model=(self.linear_pred_atoms.module if is_parallel(self.linear_pred_atoms) else self.linear_pred_atoms),
                        path=os.path.join(self.output_dir, self.best_model_path, 'linear_pred_atoms'),
                        code_paths=[os.path.join(self.root_level_dir, "model"),
                                    os.path.join(self.root_level_dir, "utils")])
                    mlflow.pytorch.save_model(
                        pytorch_model=(self.linear_pred_pharms.module if is_parallel(self.linear_pred_pharms) else self.linear_pred_pharms),
                        path=os.path.join(self.output_dir, self.best_model_path, 'linear_pred_pharms'),
                        code_paths=[os.path.join(self.root_level_dir, "model"),
                                    os.path.join(self.root_level_dir, "utils")])
                    # mlflow.pytorch.save_model(
                    #     pytorch_model=(self.linear_pred_amino.module if is_parallel(self.linear_pred_amino) else self.linear_pred_amino),
                    #     path=os.path.join(self.output_dir, self.best_model_path, 'linear_pred_amino'),
                    #     code_paths=[os.path.join(self.root_level_dir, "model"),
                    #                 os.path.join(self.root_level_dir, "utils")])
                    mlflow.pytorch.save_model(
                        pytorch_model=(self.linear_pred_bonds.module if is_parallel(self.linear_pred_bonds) else self.linear_pred_bonds),
                        path=os.path.join(self.output_dir, self.best_model_path, 'linear_pred_bonds'),
                        code_paths=[os.path.join(self.root_level_dir, "model"),
                                    os.path.join(self.root_level_dir, "utils")])

    def evaluate(self, split):
        self.model.eval()
        self.linear_pred_atoms.eval()
        self.linear_pred_pharms.eval()
        # self.linear_pred_amino.eval()
        self.linear_pred_bonds.eval()
        
        loss_accum = torch.tensor(0.).to(self.device)
        acc_atom_accum = torch.tensor(0.).to(self.device)
        acc_pharm_accum = torch.tensor(0.).to(self.device)
        # acc_amino_accum = torch.tensor(0.).to(self.device)
        acc_edge_accum = torch.tensor(0.).to(self.device)
        step = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dataloaders[split], desc="Eval_{}".format(split))):
                batch = batch.to(self.device)
                atom_rep,pharm_rep = self.model(batch)                # pred_atom = self.linear_pred_atoms(atom_rep[batch.masked_atom_indices])
                # loss = self.criterion(pred_atom.double(), batch.mask_atom_label[:,0])
                pred_atom = self.linear_pred_atoms(atom_rep[batch.nodes['a'].data['mask']==True])
                loss = self.criterion(pred_atom.double(), batch.nodes['a'].data['label'][batch.nodes['a'].data['mask']==True][:,0])
                acc_atom = compute_accuracy(pred_atom, batch.nodes['a'].data['label'][batch.nodes['a'].data['mask']==True][:,0])
                acc_atom_accum += acc_atom
                # acc_atom = torch.tensor(compute_accuracy(pred_atom, batch.mask_atom_label[:,0])).to(self.device)
                # acc_atom_accum += acc_atom
                
                # if self.cfg.train.mask_amino:
                #     pred_amino = self.linear_pred_amino(atom_rep[batch.nodes['a'].data['mask']==True])
                #     loss += self.criterion(pred_amino.double(), batch.nodes['a'].data['aa_label'][batch.nodes['a'].data['mask']==True])
                #     acc_amino = compute_accuracy(pred_amino, batch.nodes['a'].data['aa_label'][batch.nodes['a'].data['mask']==True])
                #     acc_amino_accum += acc_amino
                
                if self.cfg.train.mask_pharm:
                    pred_pharm = self.linear_pred_pharms(pharm_rep[batch.nodes['p'].data['mask']==True])
                    loss += self.criterion(pred_pharm.double(), batch.nodes['p'].data['label'][batch.nodes['p'].data['mask']==True])
                    acc_pharm = compute_accuracy(pred_pharm, batch.nodes['p'].data['label'][batch.nodes['p'].data['mask']==True])
                    acc_pharm_accum += acc_pharm
                
                if self.cfg.train.mask_edge:
                    masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
                    edge_rep = atom_rep[masked_edge_index[0]] + atom_rep[masked_edge_index[1]]
                    pred_edge = self.linear_pred_bonds(edge_rep)
                    loss += self.criterion(pred_edge.double(), batch.mask_edge_label[:,0])

                    acc_edge = torch.tensor(compute_accuracy(pred_edge, batch.mask_edge_label[:,0])).to(self.device)
                    acc_edge_accum += acc_edge
                    
                loss_accum += torch.tensor(loss.item()).to(self.device)
                step = step
                
            if self.cfg.mode.ddp:
                torch.distributed.barrier()
                torch.distributed.all_reduce(loss_accum, op=ReduceOp.SUM)
                torch.distributed.all_reduce(acc_atom_accum, op=ReduceOp.SUM)
                torch.distributed.all_reduce(acc_pharm_accum, op=ReduceOp.SUM)
                # torch.distributed.all_reduce(acc_amino_accum, op=ReduceOp.SUM)
                torch.distributed.all_reduce(acc_edge_accum, op=ReduceOp.SUM)
                
                loss_accum /= torch.distributed.get_world_size()
                acc_atom_accum /= torch.distributed.get_world_size()
                acc_pharm_accum /= torch.distributed.get_world_size()
                # acc_amino_accum /= torch.distributed.get_world_size()
                acc_edge_accum /= torch.distributed.get_world_size()
                
            metrics = {
                '{}_loss'.format(split): loss_accum.item()/(step+1),
                '{}_acc_atom'.format(split): acc_atom_accum.item()/(step+1),
                '{}_acc_pharm'.format(split): acc_pharm_accum.item()/(step+1),
                # '{}_acc_amino'.format(split): acc_amino_accum.item()/(step+1),
                '{}_acc_edge'.format(split): acc_edge_accum.item()/(step+1)}
            
        return metrics
    
    def run(self):
        for epoch in range(self.cfg.train.epochs):
            self.epoch = epoch
            self.train_epoch()

            # if self.epoch % 5 == 0:
            #     self.eval_epoch("train")
            
            self.eval_epoch("valid")    
            self.eval_epoch("test")


class Contextpred_Trainer(object):
    def __init__(self, 
                 cfg, 
                 global_rank, 
                 world_size, 
                 model_substruct, 
                 model_context, 
                 dataloaders, 
                 criterion, 
                 optimizer_substruct, 
                 optimizer_context, 
                 device, 
                 output_dir):
        self.cfg = cfg
        self.global_rank = global_rank
        self.world_size = world_size
        self.model_substruct = model_substruct
        self.model_context = model_context
        self.optimizer_substruct = optimizer_substruct
        self.optimizer_context = optimizer_context
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.device = device
        self.epoch = 0
        
        self.global_train_step = 0
        self.global_valid_eval_step = 0
        self.global_train_eval_step = 0
        self.global_test_eval_step = 0
        
        self.default_metric = 'accuracy'
        self.best_metric = -1
        self.best_model_path = Path(".")

        self.root_level_dir = os.getcwd()
        self.output_dir = output_dir
        
    def train_epoch(self):
        
        self.model_substruct.train()
        self.model_context.train()

        balanced_loss_accum = 0
        acc_accum = 0
        
        if self.cfg.mode.ddp:
            self.dataloaders['train'].sampler.set_epoch(self.epoch)
            
        for step, batch in enumerate(self.dataloaders['train']):
            batch = batch.to(self.device)

            # creating substructure representation
            substruct_rep = self.model_substruct(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct)[batch.center_substruct_idx]
            
            ### creating context representations
            overlapped_atom_rep = self.model_context(batch.x_context, batch.edge_index_context, batch.edge_attr_context)[batch.overlap_context_substruct_idx]

            #Contexts are represented by 
            if self.cfg.train.mode == "cbow":
                # positive context representation
                context_rep = self.pool_func(overlapped_atom_rep, batch.batch_overlapped_context, mode = self.cfg.train.context_pooling)
                # negative contexts are obtained by shifting the indicies of context embeddings
                neg_context_rep = torch.cat([context_rep[self.cycle_index(len(context_rep), i+1)] for i in range(self.cfg.train.neg_samples)], dim = 0)
                
                pred_pos = torch.sum(substruct_rep * context_rep, dim = 1)
                pred_neg = torch.sum(substruct_rep.repeat((self.cfg.train.neg_samples, 1))*neg_context_rep, dim = 1)

            elif self.cfg.train.mode == "skipgram":

                expanded_substruct_rep = torch.cat([substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(substruct_rep))], dim = 0)
                pred_pos = torch.sum(expanded_substruct_rep * overlapped_atom_rep, dim = 1)

                #shift indices of substructures to create negative examples
                shifted_expanded_substruct_rep = []
                for i in range(self.cfg.train.neg_samples):
                    shifted_substruct_rep = substruct_rep[self.cycle_index(len(substruct_rep), i+1)]
                    shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(shifted_substruct_rep))], dim = 0))

                shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim = 0)
                pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_atom_rep.repeat((self.cfg.train.neg_samples, 1)), dim = 1)

            else:
                raise ValueError("Invalid mode!")

            loss_pos = self.criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
            loss_neg = self.criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

            
            self.optimizer_substruct.zero_grad()
            self.optimizer_context.zero_grad()

            loss = loss_pos + self.cfg.train.neg_samples*loss_neg
            loss.backward()
            #To write: optimizer
            self.optimizer_substruct.step()
            self.optimizer_context.step()

            balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
            acc_accum += 0.5 * (float(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos) + float(torch.sum(pred_neg < 0).detach().cpu().item())/len(pred_neg))

            if (self.global_train_step+1) % self.cfg.logger.log_per_steps == 0:
                Logger.info("train | epoch: {:d}  step: {:d} | loss: {:.4f}".format(
                    self.epoch, self.global_train_step, balanced_loss_accum/(step+1)))
                
                if not self.cfg.mode.nni and self.global_rank == 0 and self.cfg.logger.log:
                    mlflow.log_metric("train/balanced_loss", balanced_loss_accum/(step+1), step=self.global_train_step)
                    mlflow.log_metric("train/accuracy", acc_accum/(step+1), step=self.global_train_step)
                
            self.global_train_step += 1

    def eval_epoch(self, split):

        metrics = self.evaluate(split)

        if split == "test":
            self.global_test_eval_step += 1
            step = self.global_test_eval_step
        elif split == "valid":
            self.global_valid_eval_step += 1
            step = self.global_valid_eval_step
        elif split == "train":
            self.global_train_eval_step += 1
            step = self.global_test_eval_step

        if self.global_rank == 0:

            for metric_name, metric_v in metrics.items():
                if isinstance(metric_v,  (float, np.float, int, np.int)):
                    metric_v = round(metric_v,5)
                elif isinstance(metric_v,  str):
                    metric_v = "\n" + metric_v
                Logger.info("{} |  step: {} | {}: {}".format(split, step, metric_name, metric_v))
                    

            if not self.cfg.mode.nni and self.global_rank == 0 and self.cfg.logger.log:
                for metric_name, metric_v in metrics.items():
                    if isinstance(metric_v,  (float, np.float, int, np.int)):
                        mlflow.log_metric(key="{}/{}".format(split, metric_name),
                                            value=metric_v,
                                            step=step)
                    elif isinstance(metric_v, str):
                        mlflow.log_text(metric_v, "{}/{}_report.txt".format(split, step))

        if split == "valid":

            # learning rate scheduler
            # if self.cfg.train.lr_scheduler.type in ("plateau",):
            #     self.scheduler.step(metrics["{}_{}".format(split, self.default_metric)])
                
            # report intermediate result for nni
            if self.cfg.mode.nni:
                metrics["default"] = metrics["{}_{}".format(split, self.default_metric)]
                nni.report_intermediate_result(metrics)

            if metrics["{}_{}".format(split, self.default_metric)] >= self.best_metric:
                self.best_metric = metrics["{}_{}".format(split, self.default_metric)]

                self.best_model_path = Path("model_step_{}_{}_{}".format(
                    self.global_valid_eval_step, self.default_metric, round(metrics["{}_{}".format(split, self.default_metric)], 3)))

                if self.global_rank == 0 and self.best_model_path.exists():
                    shutil.rmtree(self.best_model_path)

                if self.global_rank == 0 and self.cfg.logger.log:
                    mlflow.pytorch.save_model(
                        pytorch_model=(self.model_substruct.module if is_parallel(self.model_substruct) else self.model_substruct),
                        path=os.path.join(self.output_dir, self.best_model_path, 'model_substruct'),
                        code_paths=[os.path.join(self.root_level_dir, "model"),
                                    os.path.join(self.root_level_dir, "utils")])
                    mlflow.pytorch.save_model(
                        pytorch_model=(self.model_context.module if is_parallel(self.model_context) else self.model_context),
                        path=os.path.join(self.output_dir, self.best_model_path, 'model_context'),
                        code_paths=[os.path.join(self.root_level_dir, "model"),
                                    os.path.join(self.root_level_dir, "utils")])

    def evaluate(self, split):
        self.model_substruct.eval()
        self.model_context.eval()

        balanced_loss_accum = torch.tensor(0.).to(self.device)
        acc_accum = torch.tensor(0.).to(self.device)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dataloaders[split], desc="Eval_{}".format(split))):
                batch = batch.to(self.device)
                # creating substructure representation
                substruct_rep = self.model_substruct(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct)[batch.center_substruct_idx]
                ### creating context representations
                overlapped_atom_rep = self.model_context(batch.x_context, batch.edge_index_context, batch.edge_attr_context)[batch.overlap_context_substruct_idx]
                #Contexts are represented by 
                if self.cfg.train.mode == "cbow":
                    # positive context representation
                    context_rep = self.pool_func(overlapped_atom_rep, batch.batch_overlapped_context, mode = self.cfg.train.context_pooling)
                    # negative contexts are obtained by shifting the indicies of context embeddings
                    neg_context_rep = torch.cat([context_rep[self.cycle_index(len(context_rep), i+1)] for i in range(self.cfg.train.neg_samples)], dim = 0)
                    
                    pred_pos = torch.sum(substruct_rep * context_rep, dim = 1)
                    pred_neg = torch.sum(substruct_rep.repeat((self.cfg.train.neg_samples, 1))*neg_context_rep, dim = 1)

                elif self.cfg.train.mode == "skipgram":

                    expanded_substruct_rep = torch.cat([substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(substruct_rep))], dim = 0)
                    pred_pos = torch.sum(expanded_substruct_rep * overlapped_atom_rep, dim = 1)

                    #shift indices of substructures to create negative examples
                    shifted_expanded_substruct_rep = []
                    for i in range(self.cfg.train.neg_samples):
                        shifted_substruct_rep = substruct_rep[self.cycle_index(len(substruct_rep), i+1)]
                        shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat((batch.overlapped_context_size[i],1)) for i in range(len(shifted_substruct_rep))], dim = 0))

                    shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim = 0)
                    pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_atom_rep.repeat((self.cfg.train.neg_samples, 1)), dim = 1)

                else:
                    raise ValueError("Invalid mode!")

                loss_pos = self.criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
                loss_neg = self.criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())
                
                loss = loss_pos + self.cfg.train.neg_samples*loss_neg

                balanced_loss_accum += torch.tensor(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item()).to(self.device)
                acc_accum += 0.5* (torch.tensor(torch.sum(pred_pos > 0).detach().cpu().item()).to(self.device)/len(pred_pos)
                                   + torch.tensor(torch.sum(pred_neg < 0).detach().cpu().item()).to(self.device)/len(pred_neg))
        
        if self.cfg.mode.ddp:
            torch.distributed.barrier()
            torch.distributed.all_reduce(balanced_loss_accum, op=ReduceOp.SUM)
            torch.distributed.all_reduce(acc_accum, op=ReduceOp.SUM)
            
            balanced_loss_accum /= torch.distributed.get_world_size()
            acc_accum /= torch.distributed.get_world_size()

                
        metrics = {'{}_balanced_loss'.format(split): balanced_loss_accum.item()/(step+1),
                   '{}_accuracy'.format(split): acc_accum.item()/(step+1)}
        
        return metrics
    
    def run(self):
        for epoch in range(self.cfg.train.epochs):
            self.epoch = epoch
            self.train_epoch()

            # if self.epoch % 5 == 0:
            #     self.eval_epoch("train")
            
            self.eval_epoch("valid")    
            self.eval_epoch("test")
    
    def pool_func(self, x, batch, mode = "sum"):
        if mode == "sum":
            return global_add_pool(x, batch)
        elif mode == "mean":
            return global_mean_pool(x, batch)
        elif mode == "max":
            return global_max_pool(x, batch)

    def cycle_index(self, num, shift):
        arr = torch.arange(num) + shift
        arr[-shift:] = torch.arange(shift)
        return arr
                