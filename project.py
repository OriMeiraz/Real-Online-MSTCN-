from load_model import load_model, args
from batch_gen import BatchGenerator
import torch
from DataStructures import ModelRecreate, FrameGenerator
import tqdm
import time

from utils.efficientnetV2 import EfficientnetV2
from accelerate import Accelerator
from termcolor import colored
import warnings


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
        fr"{output}/experiment_20220530/{fold}/2355/model_50.pth"))
    extractor = extractor.eval()
    extractor = extractor.to(device)
    print(colored("loaded feature extraction", 'green'))
    if use_accelerator:
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
            warnings.simplefilter("ignore")
            extractor = torch.jit.trace(
                extractor, sample_input).to(device)
            extractor = torch.jit.freeze(extractor)
        for _ in range(num_examples):
            extractor(f)
        print(colored("end examples of feature extraction", "blue"))
    return extractor


def run(video_path, model, extractor, mean, std):
    outputs = []
    frame_gen = FrameGenerator(video_path)
    print(colored("initialize Model recreate - ready to start straming", "yellow"))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mr = ModelRecreate(model, model.window_dim,
                           frame_gen, extractor, mean, std)
    print(colored(
        f"finished initializing Model recreate, took {time.time()-t0} seconds", "yellow"))
    t0 = time.time()
    t = 0
    while True:
        try:
            outputs.append(mr.next())
            t += 1
            if t % 100 == 0:
                took = time.time() - t0
                print(
                    colored(f"{t} iterations in {took} seconds. That is {t/took} fps", "cyan"))

        except ValueError:
            took = time.time() - t0
            print(colored(
                f"{t} iterations in {took} seconds. That is {t/took} fps", "cyan"))
            return outputs


def main(shape, video_path):
    model, extractor = prepare_models()
    mean, std = extractor.input_mean, extractor.input_std
    extractor = extraction_examples(extractor, shape=shape)
    with torch.no_grad():
        return run(video_path, model, extractor, mean, std)


if __name__ == '__main__':
    video_path = "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon2_side"
    outputs = main(shape, video_path)
    print(outputs[n-1])
