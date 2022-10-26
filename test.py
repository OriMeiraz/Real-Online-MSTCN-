from project import device, n, prepare_models
import torch
import torch.nn.functional as F
import numpy as np
from DataStructures import ModelRecreate, RefinementLayer, PgStage, PredictionLayer
from FrameGenrator import FrameGenerator

model, extractor = prepare_models(use_accelerator=False)
frame_gen = FrameGenerator(
    "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon1_side")

batch_input = np.load(
    "/data/shared-data/scalpel/APAS-Activities/data/APAS/features/fold 2/P016_balloon1.npy")

batch_input = torch.tensor(batch_input).to(device)
batch_input = batch_input.reshape(1, *batch_input.shape)

mr = ModelRecreate(model, model.window_dim, frame_gen,
                   extractor, extractor.input_mean, extractor.input_std)
mr.next()
print(mr.next())
model.offline_mode = False
out = model(batch_input)[0][-1]
print(out[:, :, 1])

"""
pg = PgStage(model, model.window_dim, frame_gen, extractor,
             extractor.input_mean, extractor.input_std)
pl = pg.top_layer
try:
    while True:
        pl = pl.prev_layer
except:
    print('done')

pred = pg.next()
real = model.PG(batch_input, 6, True)
print(pred)
print(real)

print('hello')
"""
