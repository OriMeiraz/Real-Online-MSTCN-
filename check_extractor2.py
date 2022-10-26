from traceback import print_tb
import torch
from utils.efficientnetV2 import EfficientnetV2
from FrameGenrator import FrameGenerator
from torchvision import transforms
from utils.transforms import GroupCenterCrop, GroupScale, GroupNormalize
import random
import time


def make_deterministic(seed=None):
    """
    makes the experiment deterministic so we can replicate
    """
    if seed == None:
        seed = int(time.time())  # setting seed for deterministic output
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


make_deterministic(1)

device = torch.device('cuda')

extractor = EfficientnetV2(
    size="m", num_classes=6, pretrained=False)  # load extractor
extractor.load_state_dict(torch.load(
    "/data/shared-data/scalpel/APAS-Activities/output/experiment_20220530/2/2355/model_50.pth"))
extractor = extractor.eval().to(device)
print('extractor made')
video_path = "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon1_side"


val_augmentation = transforms.Compose([GroupScale(int(256)),
                                       GroupCenterCrop(224)])
normalize = GroupNormalize(extractor.input_mean, extractor.input_std)


fg = FrameGenerator(video_path)
img = fg.next()
print('got_image')
frame_tensor = val_augmentation([img])
frame_tensor = transforms.ToTensor()(frame_tensor[0])
frame_tensor = frame_tensor.to(device)
frame_tensor = normalize(frame_tensor)
frame_tensor = frame_tensor.view(1, *frame_tensor.size())
features = extractor(frame_tensor)[1]
print(features)
