# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology
from train_opts_2D import parser
from utils.resnet2D import resnet18
from utils.efficientnetV2 import EfficientnetV2
from utils.dataset import Gesture2dTrainSet, Sequential2DTestGestureDataSet
from utils.transforms import GroupNormalize, GroupScale, GroupCenterCrop
from utils.metrics import accuracy, average_F1, edit_score, overlap_f1
from utils.util import AverageMeter
import utils.util
import os.path
import datetime
import numpy as np
import string
import torch
import torchvision
from torch.autograd import Variable
import tqdm
import wandb
import pandas as pd
import random
"login code: 7f49a329fde9628512efec583de6188a33d0ed01"


data = r"/data/shared-data/scalpel/APAS-Activities/data/"
gesture_ids = ['G0', 'G1', 'G2', 'G3', 'G4', 'G5']
folds_folder = os.path.join(data, "APAS", "folds")


def read_data(folds_folder, split_num):
    list_of_train_examples = []
    number_of_folds = 0
    for file in os.listdir(folds_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and "fold" in filename:
            number_of_folds = number_of_folds + 1

    for file in os.listdir(folds_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and "fold" in filename:
            if str(split_num) in filename:
                file_ptr = open(os.path.join(folds_folder, filename), 'r')
                list_of_test_examples = file_ptr.read().split('\n')[:-1]
                file_ptr.close()
                random.shuffle(list_of_test_examples)
            elif str((split_num + 1) % number_of_folds) in filename:
                file_ptr = open(os.path.join(folds_folder, filename), 'r')
                list_of_examples = file_ptr.read().split('\n')[:-1]
                list_of_valid_examples = list_of_examples[0:12]
                random.shuffle(list_of_valid_examples)
                list_of_train_examples = list_of_train_examples + \
                    list_of_examples[12:]

                file_ptr.close()
            else:
                file_ptr = open(os.path.join(folds_folder, filename), 'r')
                list_of_train_examples = list_of_train_examples + \
                    file_ptr.read().split('\n')[:-1]
                file_ptr.close()
            continue
        else:
            continue

    random.shuffle(list_of_train_examples)
    return list_of_train_examples, list_of_valid_examples, list_of_test_examples


# wandb.init(name="my awesome run")

def log(msg, output_folder):
    f_log = open(os.path.join(output_folder, "log.txt"), 'a')
    utils.util.log(f_log, msg)
    f_log.close()


def eval(model, val_loaders, device_gpu, device_cpu, num_class, output_folder, gesture_ids, upload=False):
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
            log("Trial {}:\tAcc - {:.3f} Avg_F1 - {:.3f} Edit - {:.3f} F1_10 {:.3f} F1_25 {:.3f} F1_50 {:.3f}"
                .format(val_loader.dataset.video_name, acc, mean_avg_f1, edit, f1_10, f1_25, f1_50), output_folder)
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

        df = pd.DataFrame(list(zip(gesture_ids_, all_precisions, all_recalls, all_f1s)),
                          columns=['gesture_ids', 'precision', 'recall', 'f1'])
        log(df, output_folder)

        log("Overall: Acc - {:.3f} Avg_F1 - {:.3f} Edit - {:.3f} F1_10 {:.3f} F1_25 {:.3f} F1_50 {:.3f}".format(
            np.mean(overall_acc), np.mean(
                overall_avg_f1), np.mean(overall_edit),
            np.mean(overall_f1_10), np.mean(
                overall_f1_25), np.mean(overall_f1_50)
        ), output_folder)
        if upload:
            wandb.log({'validation accuracy': np.mean(overall_acc), 'Avg_F1': np.mean(overall_avg_f1), 'Edit': np.mean(
                overall_edit), "F1_10": np.mean(overall_f1_10), "F1_25": np.mean(overall_f1_25), "F1_50": np.mean(overall_f1_50)})

    model.train()
    return np.mean(overall_acc), np.mean(overall_avg_f1), results_per_vedo


def save_fetures(model, val_loaders, list_of_videos_names, device_gpu, features_path):
    video_features = []
    all_names = []

    model.eval()
    with torch.no_grad():

        for video_num, val_loader in enumerate(val_loaders):
            video_name = val_loader.dataset.video_name
            file_path = os.path.join(features_path, video_name+".npy")

            for i, batch in enumerate(val_loader):
                data, target = batch
                data = data.to(device_gpu)
                output = model(data)
                features = output[1]
                features = features.detach().cpu().numpy()
                video_features.append(features)
            print(len(video_features))
            embedding = np.concatenate(video_features, axis=0).transpose()
            np.save(file_path, embedding)
            video_features = []


def main(split=3, upload=False, save_features=False):
    data = r"/data/shared-data/scalpel/APAS-Activities/data/"

    features_path = os.path.join(
        data, "APAS", "features", "fold "+str(split))

    eval_metric = "F1"
    best_metric = 0
    best_epoch = 0
    all_eval_results = []

    print(torch.__version__)
    print(torchvision.__version__)

    # torch.backends.cudnn.enabled = False

    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        print("GPU not found - exit")
        return
    args = parser.parse_args()
    args.eval_batch_size = 2 * args.batch_size
    args.split = split
    if upload:
        wandb.init(project="New_test_proj")
        wandb.config.update(args)

    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")

    checkpoint = None
    if args.resume_exp:
        output_folder = args.resume_exp
    else:
        output_folder = os.path.join(args.out, args.exp + "_" + datetime.datetime.now().strftime("%Y%m%d"),
                                     str(split), datetime.datetime.now().strftime("%H%M"))
        os.makedirs(output_folder, exist_ok=True)

    checkpoint_file = os.path.join(output_folder, "checkpoint" + ".pth.tar")

    if args.resume_exp:
        checkpoint = torch.load(checkpoint_file)
        args_checkpoint = checkpoint['args']
        for arg in args_checkpoint:
            setattr(args, arg, args_checkpoint[arg])
        log("====================================================================", output_folder)
        log("Resuming experiment...", output_folder)
        log("====================================================================", output_folder)
    else:
        log("Used parameters...", output_folder)
        for arg in sorted(vars(args)):
            log("\t" + str(arg) + " : " + str(getattr(args, arg)), output_folder)

    args_dict = {}
    for arg in vars(args):
        args_dict[str(arg)] = getattr(args, arg)

    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if checkpoint:
        torch.set_rng_state(checkpoint['rng'])

    # ===== prepare model =====

    if args.arch == "EfficientnetV2":
        model = EfficientnetV2(
            size="m", num_classes=args.num_classes, pretrained=True)
    else:
        model = resnet18(pretrained=True, progress=True,
                         num_classes=args.num_classes)

    if checkpoint:
        # load model weights
        model.load_state_dict(checkpoint['model_weights'])

    log("param count: {}".format(sum(p.numel()
        for p in model.parameters())), output_folder)
    log("trainable params: {}".format(sum(p.numel()
        for p in model.parameters() if p.requires_grad)), output_folder)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if checkpoint:
        # load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_gpu)
    scheduler = None
    if args.use_scheduler:
        last_epoch = -1
        if checkpoint:
            last_epoch = checkpoint['epoch']
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.2, last_epoch=last_epoch)

    list_of_train_examples, list_of_valid_examples, list_of_test_examples = read_data(
        folds_folder, split)
    normalize = GroupNormalize(model.input_mean, model.input_std)
    train_augmentation = model.get_augmentation(crop_corners=args.corner_cropping,
                                                do_horizontal_flip=args.do_horizontal_flip)

    train_set = Gesture2dTrainSet(list_of_train_examples, args.data_path, args.transcriptions_dir, gesture_ids,
                                  image_tmpl=args.image_tmpl, samoling_factor=args.video_sampling_step, video_suffix=args.video_suffix,
                                  transform=train_augmentation, normalize=normalize, epoch_size=args.epoch_size, debag=False)

    def init_train_loader_worker(worker_id):
        np.random.seed(int((torch.initial_seed() + worker_id) %
                       (2**32)))  # account for randomness
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, worker_init_fn=init_train_loader_worker)
    log("Training set: will sample {} gesture snippets per pass".format(
        train_loader.dataset.__len__()), output_folder)

    test_loaders = []
    val_loaders = []

    val_augmentation = torchvision.transforms.Compose([GroupScale(int(256)),
                                                       GroupCenterCrop(args.input_size)])  # need to be corrected

    for video in list_of_valid_examples:

        data_set = Sequential2DTestGestureDataSet(root_path=args.data_path, video_id=video, transcriptions_dir=args.transcriptions_dir, gesture_ids=gesture_ids,
                                                  snippet_length=1,
                                                  sampling_step=6,
                                                  image_tmpl=args.image_tmpl,
                                                  video_suffix=args.video_suffix,
                                                  normalize=normalize,
                                                  transform=val_augmentation)  # augmentation are off
        val_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=args.eval_batch_size,
                                                       shuffle=False, num_workers=args.workers))

    for video in list_of_test_examples:
        data_set = Sequential2DTestGestureDataSet(root_path=args.data_path, video_id=video,
                                                  transcriptions_dir=args.transcriptions_dir,
                                                  gesture_ids=gesture_ids,
                                                  snippet_length=1,
                                                  sampling_step=6,
                                                  image_tmpl=args.image_tmpl,
                                                  video_suffix=args.video_suffix,
                                                  normalize=normalize,
                                                  transform=val_augmentation)  # augmentation are off
        test_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=args.eval_batch_size,
                                                        shuffle=False, num_workers=args.workers))

    if save_features is True:
        log("Start  features saving...", output_folder)

    # extract Features
        all_loaders = []
        all_videos = list_of_train_examples + \
            list_of_valid_examples + list_of_test_examples

        for video in all_videos:
            data_set = Sequential2DTestGestureDataSet(root_path=args.data_path, video_id=video,
                                                      transcriptions_dir=args.transcriptions_dir,
                                                      gesture_ids=gesture_ids,
                                                      snippet_length=1,
                                                      sampling_step=1,
                                                      image_tmpl=args.image_tmpl,
                                                      video_suffix=args.video_suffix,
                                                      normalize=normalize,
                                                      transform=val_augmentation)  # augmentation are off
            all_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=1,
                                                           shuffle=False, num_workers=args.workers))

        save_fetures(model, all_loaders, all_videos, device_gpu, features_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    main(split=0, save_features=True)
    #main(split=1, save_features=True)
    #main(split=2, save_features=True)
    #main(split=3, save_features=True)
    #main(split=4, save_features=True)
