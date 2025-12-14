import sys
import os
import torch
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from my_ultralytics.models import YOLOv10
import torch

def count_parameters(filepath):
    """
    Counts the total number of parameters in a PyTorch model file (.pt).
    This function can handle different saving formats:
    - Entire model
    - state_dict
    - Checkpoint dictionary containing 'model_state_dict' or 'model'
    """
    try:
        print(f"Loading model from {filepath}...")
        # Load the file on CPU to avoid potential CUDA issues
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        state_dict = None
        # Determine the structure of the loaded object
        if isinstance(checkpoint, dict):
            # Case 1: Checkpoint dictionary
            if 'model_state_dict' in checkpoint:
                # Common case for training checkpoints
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                # Another common case
                model_content = checkpoint['model']
                if hasattr(model_content, 'state_dict'):
                    state_dict = model_content.state_dict()
                elif isinstance(model_content, dict):
                    state_dict = model_content
                else:
                    print("Error: 'model' key found but it's not a model object or a state_dict.")
                    return None
            elif 'state_dict' in checkpoint:
                 state_dict = checkpoint['state_dict']
            else:
                # Assume the dictionary itself is the state_dict
                print("Assuming the loaded dictionary is the state_dict.")
                state_dict = checkpoint
        elif hasattr(checkpoint, 'state_dict'):
            # Case 2: Entire model was saved with torch.save(model, ...)
            print("Loaded a full model object.")
            state_dict = checkpoint.state_dict()
        else:
             # Case 3: Only state_dict was saved
            print("Loaded a state_dict object directly.")
            state_dict = checkpoint

        if state_dict is None:
            print("Could not find a valid state_dict in the file.")
            return None
            
        # Calculate total parameters
        total_params = 0
        for _, param_tensor in state_dict.items():
            if isinstance(param_tensor, torch.Tensor):
                total_params += param_tensor.numel()
        
        print("\nSuccessfully calculated parameters.")
        print(f"File: {filepath}")
        print(f"Total Parameters: {total_params:,}")
        return total_params

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Could not calculate parameters. Please check the file path and format.")
        return None

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python calculate_params.py <path_to_your_pt_file>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    count_parameters(filepath)
