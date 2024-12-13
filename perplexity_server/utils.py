import os
import sys

def get_model_name_or_path():
    # Add the parent directory to the path so that we can import the module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import LLM_MODEL_PATH
    return LLM_MODEL_PATH