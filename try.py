from utils.efficientnetV2 import EfficientnetV2
import torch
from torch import nn
import numpy as np
import time
from torchvision import transforms
from accelerate import Accelerator
from PIL import Image
# torch.jit.enable_onednn_fusion(True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = "/data/shared-data/scalpel/APAS-Activities/data/"
model = EfficientnetV2(size="s", num_classes=6, pretrained=True)
model = model.eval()
model = model.to(device)
extractor = model.base

torch.backends.cudnn.benchmark = True
img = Image.open(
    '/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P025_tissue2_side/img_00001.jpg')
img_tensor = transforms.PILToTensor()(img).to(device)/255
f = img_tensor.reshape(1, *img_tensor.shape)
sample_input = [f]


traced_model = torch.jit.trace(extractor, sample_input).to(device)
traced_model = torch.jit.freeze(traced_model)

t0 = time.time()
for _ in range(10):
    with torch.no_grad():
        img = Image.open(
            '/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P025_tissue2_side/img_00001.jpg')
        img_tensor = transforms.PILToTensor()(img).to(device)/255
        f = img_tensor.reshape(1, *img_tensor.shape)
        f = traced_model(f)
print((time.time()-t0)/10)

t0 = time.time()


img = Image.open(
    '/data/shared-data/scalpel/APAS-Activities/data/APAS/frames/P025_tissue2_side/img_00001.jpg')


with torch.no_grad():
    img_tensor = transforms.PILToTensor()(img).to(device)/255
    f = img_tensor.reshape(1, *img_tensor.shape)
    print(traced_model(f))
    print(model.base(f))

print(time.time() - t0)
print(f.shape)
