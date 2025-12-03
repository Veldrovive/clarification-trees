from typing import List
from PIL import Image
from clarification_trees.dialog_tree import DialogTrajectory

class GenerationHistory:
    def __init__(self, dialog):
        pass

class ModelInterface:
    model_name: str
    
    def generate(self, trajectory: DialogTrajectory):
        raise NotImplementedError