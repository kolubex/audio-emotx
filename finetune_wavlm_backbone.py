from dataloaders.mg_for_wavlm_backbone import dastaset_wavlm_backbone
from models.wavlm_finetuning_backbone import finetune_WavLM_backbone
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
from utils.AverageMeter import AverageMeter
from utils.train_eval_utils import set_seed, save_config, train, evaluate

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
import utils.mg_utils as utils
import os


# LoRA
from peft import LoraConfig, get_peft_model

# DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

class trainer(object):
    """
    Trainer class for finetuning RoBERTa for multi-label emotion recognition on the MovieGraphs dataset.
    """
    def __init__(self, config):
        """
        Initializes the trainer class with the config file.

        Args:
            config (dict): The config file with all the hyperparameters.
        """
        set_seed(config["seed"])
        self.config = config
        self.lr = config["lr"]
        data_split = utils.read_train_val_test_splits(self.config["resource_path"])
        self.train_dataset = dastaset_wavlm_backbone(self.config, data_split["train"], "train")
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=config["world_size"], rank=config["rank"], shuffle=False, drop_last=False)
        self.emo2id = self.train_dataset.get_emo2id_mapping()
        self.val_dataset = dastaset_wavlm_backbone(self.config, data_split["val"], "val", self.emo2id)
        val_sampler = DistributedSampler(self.val_dataset, num_replicas=config["world_size"], rank=config["rank"], shuffle=False, drop_last=False)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.config["batch_size"],
                                           num_workers=self.config['num_cpus'],
                                           collate_fn=self.train_dataset.collate,
                                           sampler=train_sampler,
                                           drop_last=False)
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.config["batch_size"],
                                         num_workers=self.config['num_cpus'],
                                         collate_fn=self.val_dataset.collate,
                                         sampler=val_sampler,
                                         drop_last=False)
        # self.device = torch.device("cuda:{}".format(self.config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:{}".format(config["rank"]))
        # self.device = torch.device("cpu")
        # self.config["device"] = "cpu"
        self.save_path = Path(self.config["save_path"])/Path(self.config["audio_feat_type"])
        # if self.save_path directory exists remove it and create a new one
        if self.save_path.exists():
            os.system("rm -rf {}".format(self.save_path))
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.epochs = self.config["epochs"]
        config["wavlm_finetune"]["top_k"] = self.train_dataset.top_k
        # self.model = finetune_WavLM_backbone(config["wavlm_finetune"], self.config["hugging_face_cache_path"]).to(self.device).to(rank)
        # self.model = DDP(self.model, device_ids=[rank],find_unused_parameters=True)
        self.config["model_name"] = f"WavLM_finetuned_backbone_t{self.train_dataset.top_k}"
        # config["model_name"] = f"WavLM_finetuned_backbone_t{obj.train_dataset.top_k}_lr1_{config['wavlm_finetune']['lr1']}_lr2_{config['wavlm_finetune']['lr2']}_{config['num_wavlm_layers']}_layers"
        

    def detach_tensors(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        elif isinstance(obj, list):
            return [self.detach_tensors(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.detach_tensors(value) for key, value in obj.items()}
        else:
            print("Undetached tensors")
            return obj

    def print_trainable_parameters(self):
        """
            Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def create_optimizer(self):
        """
            If it is wavlm finetuning, sets different lr for different layers,
            else returns Adam with same lr.
            Args:
                None
            Returns:
                optimizer (torch.optim.Adam): Adam optimizer with lr
        """
        if(self.config["wavlm_finetuning"]) and False:
            # Parameters of pretrained WavLM model
            param_groups1 = [{'params': self.model.module.model.encoder.layers[12 - self.config["num_wavlm_layers"]:12].parameters(), 'lr': self.config["wavlm_finetune"]["lr1"]}]
            
            # Parameters of all other layers
            param_groups2 = [{'params': [p for n, p in self.model.module.named_parameters(recurse=False) if "model" not in n], 'lr': self.config["wavlm_finetune"]["lr2"]}]
            
            # Combine the parameter groups
            param_groups = param_groups1 + param_groups2
            optimizer = optim.AdamW(param_groups,weight_decay=0.01)
        else:
            optimizer = optim.Adam(self.model.module.parameters(),lr=self.config["lr"])
        
        return optimizer
    
    def setup_training(self, rank, world_size):
        """
        Triggers the training process. This function is not called within the class.
        Wandb is initialized if wandb logging is enabled.
        Optimizer, scheduler and criterion are initialized.
        lr is fixed to 1e-6 for finetuning RoBERTa.
        A train method is called which trains the model.
        """
        setup(rank, world_size)
        self.model = finetune_WavLM_backbone(self.config["wavlm_finetune"], self.config["hugging_face_cache_path"]).to(self.device).to(rank)
        wavlm_config = self.config["wavlm_finetune"]
        n = wavlm_config["num_wavlm_layers"]
        lora_config = LoraConfig(
                r = wavlm_config["lora"]["r"],
                    lora_alpha = wavlm_config["lora"]["alpha"],
                    target_modules = [wavlm_config["lora"]["target_modules"][0].replace('*', str(11 - 1 - i)) for i in range(n)]+[wavlm_config["lora"]["target_modules"][1].replace('*', str(11 - 1 - i)) for i in range(n) ],
                    lora_dropout = wavlm_config["lora"]["dropout"],
                    modules_to_save = wavlm_config["lora"]["modules_to_save"],
                )
        lora_model = get_peft_model(self.model, lora_config)
        self.model = lora_model
        self.print_trainable_parameters()
        print("Lora model created")
        self.model = DDP(self.model, device_ids=[rank],find_unused_parameters=True)
        if self.config["wandb"]["logging"]:
            wandb.init(project=self.config["wandb"]["project"], entity=self.config["wandb"]["entity"])
            wandb.run.name = self.config["model_name"]
        self.optimizer = self.create_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5, verbose=True)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = torch.device("cuda:{}".format(rank))
        train(epochs=self.config['epochs'], num_labels=self.train_dataset.top_k,
              train_dataloader=self.train_dataloader, val_dataloader=self.val_dataloader,
              device=self.device, emo2id=self.emo2id, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
              criterion=self.criterion, pred_thresh=self.config["target_prediction_threshold"], masking=True,
              wandb_logging=self.config["wandb"]["logging"], model_name=self.config["model_name"],
              save_path=Path(self.save_path))
        wandb.finish()

def get_config():
    """
    Loads the config file and overrides the config file with the command line arguments.
    """
    base_conf = OmegaConf.load("config_finetune_audio.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)


if __name__ == "__main__":
    config = get_config()
    # print("Current config: {}".format(config["audio_feat_type"]))
    config["wavlm_finetuning"] = True
    # print("Current config: {}".format(config))
    save_config(config, Path(config["dumps_path"]), config["model_name"]+"__test_config.yaml")
    world_size = config["world_size"]
    # rank = config["rank"]
    obj = trainer(config)
    mp.spawn(obj.setup_training, args=(world_size,), nprocs=world_size)
    # obj.setup_training()
