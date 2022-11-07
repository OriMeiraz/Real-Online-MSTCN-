from genericpath import exists
from FrameGenrator import FrameGenerator
import project
import pickle
import os
import torch
import numpy as np
from project import device
import cv2
import tqdm


def write(img, curr_pred, curr_label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, f'with online: {curr_pred}',
                      (5, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    img = cv2.putText(img, f'regular:     {curr_label}',
                      (5, 60), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


video_path = "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon2_side"

model, extractor = project.prepare_models(use_accelerator=False)
model = model.eval()
if not os.path.exists('frames_preds'):
    mean, std = extractor.input_mean, extractor.input_std
    extractor = project.extraction_examples(extractor, shape=project.shape)
    with torch.no_grad():
        preds = project.run(video_path, model.to(
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
    model.offline_mode = False

    labels = model.to(device)(batch_input)

    with open('frames_labels', "wb") as fp:
        pickle.dump(labels, fp)

else:
    with open("frames_labels", "rb") as fp:
        labels = pickle.load(fp)

out = labels[0][-1]

for i in tqdm.tqdm(range(out.size(2))):
    img = cv2.imread(f'{video_path}/img_{str(i+1).zfill(5)}.jpg')
    curr_pred = [round(x.item(), 3) for x in preds[i][0][0]]
    curr_label = [round(x.item(), 3) for x in out[:, :, i][0]]
    img = write(img, curr_pred, curr_label)
    cv2.imwrite(f'video/img_{str(i+1).zfill(5)}.jpg', img)


print("hello")
