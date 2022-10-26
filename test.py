from project import device, n, prepare_models
import torch
import torch.nn.functional as F
import numpy as np
from DataStructures import ModelRecreate
from FrameGenrator import FrameGenerator

model, extractor = prepare_models(use_accelerator=False)
frame_gen = FrameGenerator(
    "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon2_side")
mr = ModelRecreate(model, model.window_dim, frame_gen, extractor,
                   extractor.input_mean, extractor.input_std)

batch_input = np.load(
    "/data/shared-data/scalpel/APAS-Activities/data/APAS/features/fold 2/P016_balloon2.npy")

batch_input = torch.tensor(batch_input).to(device)
batch_input = batch_input.reshape(1, *batch_input.shape)
print(batch_input.shape)

outs = model.to(device)(batch_input)
print(outs[0][-1][:, :, 0])
print(mr.next())
print('done')
