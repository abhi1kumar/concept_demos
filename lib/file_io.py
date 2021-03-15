"""
    File Input Output Operations for different kinds of files.
"""
import torch

def load_model(model, path):
    print("\n=> Loading model {}".format(path))
    model.load_state_dict(torch.load(path))
    return model

def save_model(model, path):
    print("\n=> Saving model {}".format(path))
    torch.save(model.state_dict(), path)