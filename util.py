import sys
import os
import torch

import shutil

def delete_dot_folders(root_path="."):
    for root, dirs, files in os.walk(root_path, topdown=False):
        for d in dirs:
            if d.startswith("."):
                folder_path = os.path.join(root, d)
                print(f"Deleting: {folder_path}")
                shutil.rmtree(folder_path, ignore_errors=True)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]




class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        
        
def printSet(set_str):
    set_str = str(set_str)
    num = len(set_str)
    print("="*num*3)
    print(" "*num + set_str)
    print("="*num*3)