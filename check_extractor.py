import project
import numpy as np
import torch
from FrameGenrator import FrameGenerator
from utils.transforms import GroupNormalize, GroupScale, GroupCenterCrop
from torchvision import transforms

device = torch.device('cuda')
video_path = "/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P016_balloon1_side"


batch_input = np.load(
    "/data/shared-data/scalpel/APAS-Activities/data/APAS/features/fold 2/P016_balloon1.npy")
batch_input = torch.tensor(batch_input).to(device)
batch_input = batch_input.reshape(1, *batch_input.shape)
print(batch_input.shape)

_, extractor = project.prepare_models(use_accelerator=False)
fg = FrameGenerator(video_path)

val_augmentation = transforms.Compose([GroupScale(int(256)),
                                       GroupCenterCrop(224)])
normalize = GroupNormalize(extractor.input_mean, extractor.input_std)
img = fg.next()

frame_tensor = val_augmentation([img])
frame_tensor = transforms.ToTensor()(frame_tensor[0])
frame_tensor = frame_tensor.to(device)
frame_tensor = normalize(frame_tensor)
frame_tensor = frame_tensor.view(1, *frame_tensor.size())
features = extractor(frame_tensor)[1]
print(f'pred features: \n{features}')
print(f'real features (saved as .npy): \n{batch_input[:, :, 0]}')
print(torch.norm(features - batch_input[:, :, 0]))
