import torch

def load_model(model, path:str):
    model.load_state_dict(torch.load(path))
    return model
