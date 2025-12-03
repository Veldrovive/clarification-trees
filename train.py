import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from clarification_trees.harness import DialogTreeHarness

@hydra.main(config_path="src/clarification_trees/conf", config_name="config")
def train(cfg: DictConfig):
    harness = DialogTreeHarness(cfg)

    wandb.init(
        project="dialog_trees_test",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )