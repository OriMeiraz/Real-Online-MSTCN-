from genericpath import exists
from FrameGenrator import FrameGenerator
import project
import pickle
import os
import torch
import numpy as np
from project import device

model, extractor = project.prepare_models(use_accelerator=False)
model = model.eval()
if not os.path.exists('frames_preds'):
    mean, std = extractor.input_mean, extractor.input_std
    extractor = project.extraction_examples(extractor, shape=project.shape)
    with torch.no_grad():
        preds = project.run(project.video_path, model.to(
            project.device), extractor, mean, std)
    with open('frames_preds', "wb") as fp:
        pickle.dump(preds, fp)

else:
    with open("frames_preds", "rb") as fp:
        preds = pickle.load(fp)


if not os.path.exists('frames_label'):
    batch_input = np.load(
        "/data/shared-data/scalpel/APAS-Activities/data/APAS/features/fold 2/P016_balloon2.npy")
    batch_input = torch.tensor(batch_input).to(device)
    batch_input = batch_input.reshape(1, *batch_input.shape)
    print(batch_input.shape)

    labels = model.to(device)(batch_input)

    with open('frames_labels', "wb") as fp:
        pickle.dump(labels, fp)

else:
    with open("frames_labels", "rb") as fp:
        labels = pickle.load(fp)


print(preds[0])
