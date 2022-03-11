import torch
import numpy as np
mask_file = "./prune0.2/mask.pth"
masks = torch.load(mask_file)

logfilename = "mask.log"
logfile = open(logfilename, "w")

for key, item in masks.items():
    print(key)
    w_mask = item["weight"]
    shape = w_mask.size()
    count = np.prod(shape[1:])
    all_ones = (w_mask.flatten(1).sum(-1) == count).nonzero().squeeze(1).tolist()
    all_zeros = (w_mask.flatten(1).sum(-1) == 0).nonzero().squeeze(1).tolist()
    logfile.write(key+"\n")
    logfile.write(str(len(all_ones)))
    logfile.write(str(all_ones)+"\n")
    logfile.write(str(len(all_zeros)))
    logfile.write(str(all_zeros)+"\n")
    pass
