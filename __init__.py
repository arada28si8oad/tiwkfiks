# __init__.py

# This file tells Python how to import the nodes from your script.

# Import the node classes from your file
from .flux_kontext_lora_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Expose the mappings to ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']