from typing import List, Optional
from PIL import Image
from clarification_trees.dialog_tree import DialogTrajectory
from enum import Enum



class ModelType(Enum):
    CLARIFICATION = "clarification"
    ANSWER = "answer"

class ModelInterface:
    model_name: str
    model_type: ModelType
    
    def generate(self, trajectory: DialogTrajectory, base_prompt_override: Optional[str] = None):
        raise NotImplementedError