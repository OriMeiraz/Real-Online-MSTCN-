from project import device, n, prepare_models
import torch
import torch.nn.functional as F
import numpy as np

model, _ = prepare_models()
batch_input = np.load(
    "/data/shared-data/scalpel/APAS-Activities/data/APAS/features/fold 2/P016_balloon2.npy")

batch_input = torch.tensor(batch_input).to(device)
batch_input = batch_input.reshape(1, *batch_input.shape)
print(batch_input.shape)

outs = model(batch_input)
out = torch.cat(outs, 2)[-1]
print(out[:, :, n-1])
