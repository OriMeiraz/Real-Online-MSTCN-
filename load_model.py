import torch
from model import MST_TCN2, MST_TCN2_early, MST_TCN2_late
import random
import argparse
import time
import os
from datetime import datetime


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


def get_args():
    """
    get the arguments of the run from the parser
    """
    dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['APAS'], default="APAS")
    parser.add_argument(
        '--task', choices=['gestures', 'tools', 'multi-taks'], default="gestures")
    parser.add_argument(
        '--network', choices=['MS-TCN2', 'MS-TCN2 late', 'MS-TCN2 early'], default="MS-TCN2")
    parser.add_argument(
        '--split', choices=['0', '1', '2', '3', '4', 'all'], default='all')
    parser.add_argument('--features_dim', default=1280, type=int)
    parser.add_argument('--lr', default='0.0010351748096577', type=float)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--eval_rate', default=1, type=int)

    # Architectuyre
    parser.add_argument('--window_dim', default=6, type=int)
    parser.add_argument('--num_layers_PG', default=10, type=int)
    parser.add_argument('--num_layers_R', default=10, type=int)
    parser.add_argument('--num_f_maps', default=128, type=int)

    parser.add_argument('--normalization', choices=[
                        'Min-max', 'Standard', 'samplewise_SD', 'none'], default='none', type=str)
    parser.add_argument('--num_R', default=3, type=int)

    parser.add_argument('--sample_rate', default=1, type=int)
    parser.add_argument('--offline_mode', default=False, type=bool)

    parser.add_argument('--loss_tau', default=16, type=float)
    parser.add_argument('--loss_lambda', default=1, type=float)
    parser.add_argument('--dropout_TCN', default=0.5, type=float)
    parser.add_argument(
        '--project', default="Offline RNN nets Sensor paper Final", type=str)
    parser.add_argument('--group', default=dt_string + " ", type=str)
    parser.add_argument('--use_gpu_num', default="1", type=str)
    parser.add_argument('--upload', default=False, type=bool)
    parser.add_argument('--debagging', default=False, type=bool)
    parser.add_argument('--hyper_parameter_tuning', default=False, type=bool)

    args = parser.parse_args()

    debagging = args.debagging
    if debagging:
        args.upload = False
    return args


# parameters for the code - regardless of run
args = get_args()
experiment_name = args.group + " task:" + args.task + " splits: " + args.split + " net: " + \
    args.network + " is Offline: " + \
    str(args.offline_mode) + " window dim: " + str(args.window_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = "/data/shared-data/scalpel/APAS-Activities/data/"
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num  # number of GPUs

gt_path_gestures = os.path.join(
    data, args.dataset, "transcriptions_gestures")
gt_path_tools_left = os.path.join(
    data, args.dataset, "transcriptions_tools_left")
gt_path_tools_right = os.path.join(
    data, args.dataset, "transcriptions_tools_right")
mapping_gestures_file = os.path.join(
    data, args.dataset, "mapping_gestures.txt")

mapping_tool_file = os.path.join(data, args.dataset, "mapping_tools.txt")
model_dir = os.path.join("models", args.dataset,
                         experiment_name, "split" + args.split)


def load_model(args=args, path=None):
    """
    returns the correct model based on the args of the run
    """
    if path == None:
        if args.network == "MS-TCN2":
            if args.task == "tools":
                path = f"/data/home/ori.meiraz/models/{args.dataset}/08.09.2022 13:34:26  task:{args.task} splits: all net: MS-TCN2 is Offline: True window dim: 6/split0"
            elif args.task == "multi-taks":
                path = f"/data/home/ori.meiraz/models/{args.dataset}/08.09.2022 14:37:19  task:{args.task} splits: all net: MS-TCN2 is Offline: True window dim: 6/split0"
            else:
                path = "/data/home/ori.meiraz/models/APAS/16.08.2022 18:04:33  task:gestures splits: all net: MS-TCN2 is Offline: True window dim: 6/split2"
    num_classes_list, _, _ = get_num_classes_list(args)
    if args.network == "MS-TCN2":
        model = MST_TCN2(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps,
                         args.features_dim, num_classes_list, dropout=args.dropout_TCN, window_dim=args.window_dim, offline_mode=True)
        model.load_state_dict(torch.load(
            f"{path}/{args.network}_{args.task}.model"))

        model.eval()
        model = model.to(device)
    elif args.network == "MS-TCN2 late":
        model = MST_TCN2_late(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps,
                              args.features_dim, num_classes_list, dropout=args.dropout_TCN, offline_mode=args.offline_mode)
        model.load_state_dict(torch.load(
            f"{path}/{args.network}_{args.task}.model"))
        model.eval()
        model = model.to(device)
    elif args.network == 'MS-TCN2 early':
        model = MST_TCN2_early(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps,
                               args.features_dim, num_classes_list, dropout=args.dropout_TCN, offline_mode=args.offline_mode)
        model.load_state_dict(torch.load(
            f"{path}/{args.network}_{args.task}.model"))
        model.eval()
        model = model.to(device)
    else:
        raise NotImplemented

    for param in model.parameters():
        param.grad = None
    torch.backends.cudnn.benchmark = True
    return model


def get_action_dicts(args) -> tuple:
    """
    return the actions_dict_tools, actions_dict_gestures
    as defined in 'train_experiment'

    Args:
        args: same as everywhere

    Returns:
        actions_dict_tools, actions_dict_gestures
    """

    ## actions_dict_gestures ##
    actions_dict_gestures = dict()
    file_ptr = open(mapping_gestures_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    for a in actions:
        actions_dict_gestures[a.split()[1]] = int(a.split()[0])

    if args.dataset != "APAS":
        return dict(), actions_dict_gestures
    ###########################
    ##   actions_dict_tools  ##
    actions_dict_tools = dict()
    file_ptr = open(mapping_tool_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    for a in actions:
        actions_dict_tools[a.split()[1]] = int(a.split()[0])
    return actions_dict_tools, actions_dict_gestures
    ###########################


def get_num_classes_list(args):
    actions_dict_tools, actions_dict_gestures = get_action_dicts(args)

    num_classes_tools = 0
    if args.dataset == "APAS":
        num_classes_tools = len(actions_dict_tools)
    num_classes_gestures = len(actions_dict_gestures)

    if args.task == "gestures":
        num_classes_list = [num_classes_gestures]
    elif args.dataset == "APAS" and args.task == "tools":
        num_classes_list = [num_classes_tools, num_classes_tools]
    elif args.dataset == "APAS" and args.task == "multi-taks":
        num_classes_list = [num_classes_gestures,
                            num_classes_tools, num_classes_tools]
    return num_classes_list, num_classes_gestures, num_classes_tools
