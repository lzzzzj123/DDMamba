import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from MambaMRI.MambaMRI_structure.models.MID_net import IDNet as Model

from utils_fvcore import FLOPs
fvcore_flop_count = FLOPs.fvcore_flop_count

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    H=256
    W=256
    scale=1
    init_model = Model().to(device)
    with torch.no_grad():
        FLOPs.fvcore_flop_count(init_model, input_shape=(1, H//scale,W//scale))

    print(sum(p.numel() for p in init_model.parameters() if p.requires_grad))

        