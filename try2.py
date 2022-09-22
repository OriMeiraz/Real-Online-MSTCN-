from PIL import Image
import numpy as np
import torch
import torchvision
from utils.efficientnetV2 import EfficientnetV2
from utils.transforms import GroupNormalize, GroupScale, GroupCenterCrop
from utils.train_opts_2D import parser
from utils.dataset import Gesture2dTrainSet, Sequential2DTestGestureDataSet
from utils.metrics import accuracy, average_F1, edit_score, overlap_f1


def eval(model, val_loaders, gesture_ids,  device_gpu='cuda', device_cpu='cpu', num_class=6, upload=False):
    results_per_vedo = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    model.eval()
    with torch.no_grad():

        overall_acc = []
        overall_avg_f1 = []
        overall_edit = []
        overall_f1_10 = []
        overall_f1_25 = []
        overall_f1_50 = []
        for video_num, val_loader in enumerate(val_loaders):
            P = np.array([], dtype=np.int64)
            Y = np.array([], dtype=np.int64)
            for i, batch in enumerate(val_loader):
                data, target = batch
                Y = np.append(Y, target.numpy())
                data = data.to(device_gpu)
                output = model(data)
                if model.arch == "EfficientnetV2":
                    output = output[0]

                predicted = torch.nn.Softmax(dim=1)(output)
                _, predicted = torch.max(predicted, 1)

                P = np.append(P, predicted.to(device_cpu).numpy())

            acc = accuracy(P, Y)
            mean_avg_f1, avg_precision, avg_recall, avg_f1 = average_F1(
                P, Y, n_classes=num_class)
            all_precisions.append(avg_precision)
            all_recalls.append(avg_recall)
            all_f1s.append(avg_f1)

            avg_precision_ = np.array(avg_precision)
            avg_recall_ = np.array(avg_recall)
            avg_f1_ = np.array(avg_f1)
            avg_precision.append(
                np.mean(avg_precision_[(avg_precision_) != np.array(None)]))
            avg_recall.append(
                np.mean(avg_recall_[(avg_recall_) != np.array(None)]))
            avg_f1.append(np.mean(avg_f1_[(avg_f1_) != np.array(None)]))
            edit = edit_score(P, Y)
            f1_10 = overlap_f1(P, Y, n_classes=num_class, overlap=0.1)
            f1_25 = overlap_f1(P, Y, n_classes=num_class, overlap=0.25)
            f1_50 = overlap_f1(P, Y, n_classes=num_class, overlap=0.5)
            results_per_vedo.append(
                [val_loader.dataset.video_name, acc, mean_avg_f1])

            overall_acc.append(acc)
            overall_avg_f1.append(mean_avg_f1)
            overall_edit.append(edit)
            overall_f1_10.append(f1_10)
            overall_f1_25.append(f1_25)
            overall_f1_50.append(f1_50)

        gesture_ids_ = gesture_ids.copy() + ["mean"]
        all_precisions = np.array(all_precisions).mean(0)
        all_recalls = np.array(all_recalls).mean(0)
        all_f1s = np.array(all_f1s).mean(0)

    return np.mean(overall_acc), np.mean(overall_avg_f1), results_per_vedo


args = parser.parse_args()
data = r"/data/shared-data/scalpel/APAS-Activities/data/APAS/"
args.data_path = data+'frames'
args.transcriptions_dir = fr"{data[:-6]}{args.transcriptions_dir[4:]}"
print(args.transcriptions_dir)
print("\n\n\n\n\n\n\n\n\n\n\n\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output = r"/data/shared-data/scalpel/APAS-Activities/output/"

check = "P016_balloon1"
fold = 2

val_augmentation = torchvision.transforms.Compose([GroupScale(int(256)),
                                                   GroupCenterCrop(args.input_size)])


model = EfficientnetV2(size="m", num_classes=6, pretrained=False)
model.load_state_dict(torch.load(
    fr"{output}/experiment_20220530/{fold}/2355/model_50.pth"))
model = model.eval()
model = model.to(device)

img = Image.open(fr"{data}/frames/{check}_side/img_04450.jpg")
a = val_augmentation([img])


gesture_ids = ['G0', 'G1', 'G2', 'G3', 'G4', 'G5']
normalize = GroupNormalize(model.input_mean, model.input_std)

aug_img = val_augmentation([img])
aug_img = [torchvision.transforms.ToTensor()(img) for img in aug_img]
snippet = aug_img[0].to(device)
snippet = normalize(snippet)
should = model(snippet.reshape(1, *snippet.shape))
should = should[1]

features = np.load(fr"{data}/features/fold {fold}/{check}.npy")
np.save("features.npy", features)
features = np.load("features.npy")
print(np.linalg.norm(features[:, 4449] -
      should.reshape(-1).cpu().detach().numpy()))
