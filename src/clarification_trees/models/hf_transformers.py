from typing import List
import torch
from transformers import AutoProcessor
from clarification_trees.models.model_interface import ModelInterface
from pathlib import Path
from omegaconf import DictConfig
from typing import Optional

from clarification_trees.dialog_tree import DialogTrajectory

class TransformersModel(ModelInterface):
    """
    Loads a model from the transformers library and provides a unified interface for tokenizing and generating text.
    """
    def __init__(self, model_config: DictConfig, device: str, shared_model: Optional['TransformersModel'] = None):
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.device = device
        self.shared_model = shared_model

        self.max_new_tokens = model_config.max_new_tokens

        self._load_model(self.model_config, shared_model=shared_model)
    
    def _load_qwen_vl_model(self, model_config):
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            raise ImportError("Qwen3VLForConditionalGeneration is not available. Please install transformers.")
        
        if model_config.use_flash_attention:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_config.model_hf_transformers_key,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=self.device,
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_config.model_hf_transformers_key, dtype="auto", device_map=self.device
            )
        self.processor = AutoProcessor.from_pretrained(model_config.model_hf_transformers_key)
    
    def _load_model(self, model_config: DictConfig, shared_model: Optional['TransformersModel'] = None):
        if model_config.model_name == "qwen-3-vl-2b":
            self._load_qwen_vl_model(model_config)
        elif model_config.model_name == "qwen-3-vl-4b":
            self._load_qwen_vl_model(model_config)
        elif model_config.model_name == "qwen-3-vl-8b":
            self._load_qwen_vl_model(model_config)
        elif model_config.model_name == "shared_model":
            self.model = shared_model.model
            self.processor = shared_model.processor
        else:
            raise NotImplementedError(f"Model {model_config.model_name} is not implemented")

    # def preprocess_text(self, text: List[str]):
    #     pass

    # def preprocess_images(self, images: List[Image.Image]):
    #     pass

    def preprocess_inputs(self, trajectory: DialogTrajectory, base_prompt_override: str = None):
        messages = trajectory.to_messages(model_name=self.model_name)
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": base_prompt_override or self.model_config.base_prompt}]})
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        return inputs

    def generate(self, trajectory: DialogTrajectory, base_prompt_override: Optional[str] = None):
        inputs = self.preprocess_inputs(trajectory, base_prompt_override)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return generated_text
        


if __name__ == "__main__":
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    from pathlib import Path
    import os

    config_path = Path(__file__).parent.parent / "conf"
    config_path = os.path.relpath(config_path, Path(__file__).parent)

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config")

    # Print out the config keys
    print(OmegaConf.to_yaml(cfg))

    device = cfg.training.device
    model = TransformersModel(cfg.clarification_model, device)

    # Construct a test trajectory
    from clarification_trees.dialog_tree import DialogTree, NodeType
    from pathlib import Path
    from PIL import Image
    
    image_path = Path(__file__).parent.parent.parent.parent / "data/clearvqa/images/train_000007.jpg"
    image = Image.open(image_path)
    tree = DialogTree("Is it real or a statue?", image, "real", ["real", "real"])
    cq = tree.add_node(DialogTree.ROOT, NodeType.CLARIFICATION_QUESTION, None, "Are you referring to the cat on the right or the cat statue on the left?")
    ca = tree.add_node(cq, NodeType.CLARIFYING_ANSWER, None, "The cat on the left.")
    inference = tree.add_node(ca, NodeType.INFERENCE, None, "real")
    
    input_trajectory = tree.get_trajectory(tree.ROOT)
    print(input_trajectory)

    generated_text = model.generate(input_trajectory)
    print(generated_text)
    