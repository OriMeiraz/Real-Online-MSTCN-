from try1 import arr as gt_results
from FrameGenrator import FrameGenerator
from DataStructures import ModelRecreate
from project import prepare_models, extraction_examples, device, shape
import torch
from project import main
from visualize_gestures import process_video
from try2 import to_vid
import pickle
import os
import math

video_path = "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon2_side/"
gesture_to_name = {'G0': "no gesture",
                   'G1': "needle passing",
                   'G2': "pull the suture",
                   'G3': "Instrument tie",
                   'G4': "Lay the knot",
                   'G5': "Cut the suture"}
actions_dict = {'G0': 0,
                'G1': 1,
                'G2': 2,
                'G3': 3,
                'G4': 4,
                'G5': 5}


rev_actions = dict([(v, k) for k, v in actions_dict.items()])
rev_g2n = dict([(v, k) for k, v in gesture_to_name.items()])

gt_results = [torch.tensor(actions_dict[rev_g2n[i]]) for i in gt_results]

if False:
    to_vid("/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon2_side/",
           'adam.avi', 0)


else:
    if not os.path.exists('adam_results_inf'):
        model, extractor = prepare_models()
        mean, std = extractor.input_mean, extractor.input_std
        extractor = extraction_examples(extractor, shape=shape)
        frame_gen = FrameGenerator(
            "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon2_side/")

        mr = ModelRecreate(model.to(device), model.window_dim,
                           frame_gen, extractor.to(device), mean, std)
        results = []
        for _ in range(3910):
            results.append(torch.argmax(mr.next()[0]).item())
        with open("adam_results_inf", "wb") as fp:  # Pickling
            pickle.dump(results, fp)
    else:
        with open("adam_results_inf", "rb") as fp:   # Unpickling
            results = pickle.load(fp)

    results = [rev_actions[i] for i in results]
    print('start making video')
    process_video('adam.avi', results, gt_results,
                  'labaled_adam_inf', gesture_to_name, actions_dict)
    print("hello")
