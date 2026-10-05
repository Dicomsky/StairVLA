import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


def _as_int_list(values):
    return [int(value) for value in values]


def _sync_action_indices_from_action_model(cfg):
    """Align dataset action labels with framework.action_model settings."""
    vla_dataset_cfg = cfg.datasets.vla_data
    action_cfg = cfg.framework.get("action_model", None)
    if action_cfg is None:
        return

    dataset_indices = vla_dataset_cfg.get("action_indices", None)
    model_indices = action_cfg.get("action_indices", None)
    default_horizon = action_cfg.get("action_horizon", None)

    if dataset_indices is not None:
        return
    elif model_indices is not None:
        vla_dataset_cfg.action_indices = _as_int_list(model_indices)
    elif default_horizon is not None:
        vla_dataset_cfg.action_indices = list(range(int(default_horizon)))


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        _sync_action_indices_from_action_model(cfg)

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        def _as_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        num_workers = int(getattr(cfg.datasets.vla_data, "num_workers", 4))
        pin_memory = _as_bool(getattr(cfg.datasets.vla_data, "pin_memory", False))
        dataloader_kwargs = dict(
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # shuffle=True
        )
        if num_workers > 0:
            prefetch_factor = getattr(cfg.datasets.vla_data, "prefetch_factor", None)
            persistent_workers = getattr(cfg.datasets.vla_data, "persistent_workers", None)
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
            if persistent_workers is not None:
                dataloader_kwargs["persistent_workers"] = _as_bool(persistent_workers)

        vla_train_dataloader = DataLoader(vla_dataset, **dataloader_kwargs)
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
