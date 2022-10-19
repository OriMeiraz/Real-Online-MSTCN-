from sys import prefix
from load_model import load_model, args
import torch
from DataStructures import ModelRecreate
from FrameGenrator import FrameGenerator
import time
from utils.efficientnetV2 import EfficientnetV2
from termcolor import colored
import warnings
import tqdm
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = "/data/shared-data/scalpel/APAS-Activities/data/"


output = r"/data/shared-data/scalpel/APAS-Activities/output/"
check = "P016_balloon1"
fold = 2
shape = (224, 224)
n = 3910


def prepare_models(use_accelerator=True):
    """loads the MSTCN++ model and the EfficientNetV2 feature extraction

    Args:
        use_accelerator (bool, optional): whether to use the accelerator of "huggingface". Defaults to True.

    Returns:
        MSTCN++: the MSTCN++ model
        EfficientNetV2: The feature extraction
    """
    model = load_model(args=args)  # load the model
    model = model.to(device).eval()
    print(colored("loaded initial model", 'green'))
    extractor = EfficientnetV2(
        size="m", num_classes=6, pretrained=False)  # load extractor
    extractor.load_state_dict(torch.load(
        fr"{output}/experiment_20220530/2/2355/model_50.pth"))

    extractor = extractor.eval()
    extractor = extractor.to(device)
    print(colored("loaded feature extraction", 'green'))
    if use_accelerator:
        from accelerate import Accelerator
        accelerator = Accelerator()
        extractor, model = accelerator.prepare(extractor, model)
        print(colored("prepared extraction and model with accelerator", 'green'))
    return model, extractor


def extraction_examples(extractor, shape: tuple = None, num_examples=30):
    """
    builds a onednn model that replicates the extractor and
    preforms 'num_examples' iterations of passing a random input of shape 'shape'
    to it to make it faster. 
    Can only do it with a pre-defined shape (the shape of each frame) 

    Args:
        extractor (EfficientNetV2): The extractor
        shape (tuple, optional): Defaults to None.
        num_examples (int, optional): number of iterations as described . Defaults to 30.

    Returns:
        onednn model: as described
    """
    if shape is not None:
        print(colored("start examples of feature extraction", "blue"))
        f = torch.randn(1, 3, *shape, device=device)
        sample_input = [f]
        with warnings.catch_warnings():
            with torch.no_grad():
                extractor = extractor.to(device)
                warnings.simplefilter("ignore")
                extractor = torch.jit.trace(
                    extractor, sample_input).to(device)
                print('extracted')
                extractor = torch.jit.freeze(extractor)
        for i in range(num_examples):
            print(i)
            extractor(f)
        print(colored("end examples of feature extraction", "blue"))
    return extractor


def run(video_path, model, extractor, mean, std):
    outputs = []
    frame_gen = FrameGenerator(video_path)
    print(colored("initialize Model recreate - ready to start streaming", "yellow"))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mr = ModelRecreate(model, model.window_dim,
                           frame_gen, extractor, mean, std)
    print(colored(
        f"finished initializing Model recreate, took {time.time()-t0} seconds", "yellow"))
    pbar = tqdm.tqdm(total=math.inf, unit=' frames',
                     bar_format='Calculated {n} frames. current rate: {rate_fmt}')
    while True:
        try:
            outputs.append(mr.next())
            pbar.update(1)

        except ValueError:
            pbar.bar_format = 'Total number of {n} frames. Calculated at avg of {rate_fmt} '
            pbar.close()
            return outputs


def main(shape, video_path):
    model, extractor = prepare_models(use_accelerator=False)
    mean, std = extractor.input_mean, extractor.input_std
    extractor = extraction_examples(extractor, shape=shape)
    with torch.no_grad():
        return run(video_path, model.to(device), extractor, mean, std)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    video_path = "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon2_side"
    outputs = main(shape, video_path)
