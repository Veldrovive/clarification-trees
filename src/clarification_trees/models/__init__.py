from .model_interface import ModelInterface
from .hf_transformers import TransformersModel
from omegaconf import DictConfig
from typing import Optional
from loguru import Logger

def construct_model(model_cfg: DictConfig, logger: Optional[Logger] = None,  shared_model: Optional[ModelInterface] = None):
    if model_cfg.model_type == "huggingface":
        return TransformersModel(model_cfg, logger, shared_model)
    else:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")

def construct_models()