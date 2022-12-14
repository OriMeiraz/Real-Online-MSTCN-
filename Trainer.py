# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com

from model import *
import os
from torch import optim
import math
import pandas as pd
from termcolor import colored, cprint

from metrics import *
import wandb
from datetime import datetime
import tqdm
from scipy import signal


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_list, offline_mode=False, window_dim=0, tau=16, lambd=0.15, dropout_TCN=0.5, task="gestures", device="cuda",
                 network='MS-TCN2', hyper_parameter_tuning=False, debagging=False):
        if network == 'MS-TCN2':
            self.model = MST_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps,
                                  dim, num_classes_list, dropout=dropout_TCN, offline_mode=offline_mode)
        elif network == 'MS-TCN2 late':
            self.model = MST_TCN2_late(num_layers_PG, num_layers_R, num_R, num_f_maps,
                                       dim, num_classes_list, dropout=dropout_TCN, offline_mode=offline_mode)
        elif network == 'MS-TCN2 early':
            self.model = MST_TCN2_early(num_layers_PG, num_layers_R, num_R, num_f_maps,
                                        dim, num_classes_list, dropout=dropout_TCN, offline_mode=offline_mode)

        else:
            raise NotImplemented
        self.number_params = sum(p.numel()
                                 for p in self.model.parameters() if p.requires_grad)

        self.window_dim = window_dim
        self.model.window_dim = window_dim
        self.debagging = debagging
        self.network = network
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss()
        self.num_classes_list = num_classes_list
        self.tau = tau
        self.lambd = lambd
        self.task = task
        self.hyper_parameter_tuning = hyper_parameter_tuning

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, eval_dict, args):
        best_valid_results = None
        Max_F1_macro = 0
        number_of_seqs = len(batch_gen.list_of_train_examples)
        number_of_batches = math.ceil(number_of_seqs / batch_size)

        eval_results_list = []
        train_results_list = []
        print(args.dataset + " " + args.group + " " +
              args.dataset + " dataset " + "split: " + args.split)

        if args.upload is True:
            wandb.init(project=args.project, group=args.group,
                       name="split: " + args.split,
                       reinit=True)
            delattr(args, 'split')
            wandb.config.update(args)

        self.model.train()
        self.model.to(self.device)
        eval_rate = eval_dict["eval_rate"]
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            correct1 = 0
            total1 = 0
            correct2 = 0
            total2 = 0
            correct3 = 0
            total3 = 0

            while batch_gen.has_next():
                if self.task == "multi-taks":
                    batch_input, batch_target_left, batch_target_right, batch_target_gestures, mask_gesture, mask_tools = batch_gen.next_batch(
                        batch_size)
                    batch_input, batch_target_left, batch_target_right, batch_target_gestures, mask_gesture, mask_tools = batch_input.to(
                        self.device), batch_target_left.to(self.device), batch_target_right.to(
                        self.device), batch_target_gestures.to(self.device), mask_gesture.to(self.device), mask_tools.to(self.device)
                    mask = mask_gesture

                elif self.task == "tools":
                    batch_input, batch_target_left, batch_target_right, mask = batch_gen.next_batch(
                        batch_size)
                    batch_input, batch_target_left, batch_target_right, mask = batch_input.to(self.device), batch_target_left.to(
                        self.device), batch_target_right.to(self.device), mask.to(self.device)
                else:
                    batch_input, batch_target_gestures, mask = batch_gen.next_batch(
                        batch_size)
                    batch_input, batch_target_gestures, mask = batch_input.to(self.device), batch_target_gestures.to(
                        self.device), mask.to(self.device)

                optimizer.zero_grad()
                predictions1, predictions2, predictions3 = [], [], []
                lengths = torch.sum(mask[:, 0, :], dim=1).to(
                    dtype=torch.int64).to(device='cpu')

                if self.task == "multi-taks":

                    predictions1, predictions2, predictions3 = self.model(
                        batch_input, lengths)
                    predictions1 = (predictions1 * mask_gesture)
                    predictions2 = (predictions2 * mask_tools)
                    predictions3 = (predictions3 * mask_tools)

                elif self.task == "tools":
                    predictions2, predictions3 = self.model(
                        batch_input, lengths)
                    predictions2 = (predictions2 * mask)
                    predictions3 = (predictions3 * mask)

                else:
                    predictions1 = self.model(batch_input, lengths)
                    # the first entry in predictions1 is original prediction
                    # and the num_R refinement stages
                    predictions1 = (predictions1[0] * mask)

                loss = 0
                for p in predictions1:
                    # p[0] is the predictions of time t-1 (t is the current time)
                    # p[1] is the predictions of time t
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_list[0]),
                                    batch_target_gestures.view(-1))
                    if self.network not in ["GRU", "LSTM"]:
                        loss += self.lambd * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=self.tau))

                for p in predictions2:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_list[1]),
                                    batch_target_right.view(-1))
                    if self.network not in ["GRU", "LSTM"]:
                        loss += self.lambd * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=self.tau))

                for p in predictions3:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_list[1]),
                                    batch_target_left.view(-1))
                    if self.network not in ["GRU", "LSTM"]:
                        loss += self.lambd * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=self.tau))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                if self.task == "multi-taks" or self.task == "gestures":
                    _, predicted1 = torch.max(
                        predictions1[-1].data, 1)  # last prediction
                    for i in range(len(lengths)):

                        correct1 += (predicted1[i][:lengths[i]] ==
                                     batch_target_gestures[i][:lengths[i]]).float().sum().item()
                        total1 += lengths[i]

                if self.task == "multi-taks" or self.task == "tools":

                    _, predicted2 = torch.max(predictions2[-1].data, 1)
                    _, predicted3 = torch.max(predictions3[-1].data, 1)
                    correct2 += ((predicted2 ==
                                 batch_target_right).float().squeeze(1)).sum().item()
                    total2 += predicted2.shape[1]
                    correct3 += ((predicted3 ==
                                 batch_target_left).float().squeeze(1)).sum().item()
                    total3 += predicted3.shape[1]

                pbar.update(1)

            batch_gen.reset()
            pbar.close()
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if self.task == "multi-taks":
                print(colored(dt_string, 'green', attrs=[
                    'bold']) + "  " + "[epoch %d]: train loss = %f,  train acc gesture = %f,  train acc right= %f,  train acc left = %f" % (
                    epoch + 1,
                    epoch_loss / len(batch_gen.list_of_train_examples),
                    float(correct1) / total1, float(correct2) / total2, float(correct3) / total3))

                train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                                 "train acc left": float(correct1) / total1, "train acc right": float(correct2) / total2,
                                 "train acc gestures": float(correct3) / total3}
            elif self.task == "tools":
                print(colored(dt_string, 'green', attrs=[
                    'bold']) + "  " + "[epoch %d]: train loss = %f,   train acc right = %f,  train acc left = %f" % (
                    epoch + 1,
                    epoch_loss / len(batch_gen.list_of_train_examples),
                    float(correct2) / total2, float(correct3) / total3))
                train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                                 "train acc left": float(correct2) / total2,
                                 "train acc right": float(correct3) / total3}
            else:
                print(colored(dt_string, 'green',
                              attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
                                                                                                          epoch_loss / len(
                                                                                                              batch_gen.list_of_train_examples),
                                                                                                          float(
                                                                                                              correct1) / total1))
                train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                                 "train acc": float(correct1) / total1}

            if args.upload:
                wandb.log(train_results)

            train_results_list.append(train_results)

            if (epoch+1) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) +
                      " model evaluation", 'red', attrs=['bold']))
                results = {"epoch": epoch + 1}
                results.update(self.evaluate(eval_dict, batch_gen))
                eval_results_list.append(results)
                if self.task == "gestures":
                    if results['F1-macro gesture'] >= Max_F1_macro:
                        Max_F1_macro = results['F1-macro gesture']
                        best_valid_results = results
                        if not self.debagging and not self.hyper_parameter_tuning:
                            torch.save(self.model.state_dict(), save_dir +
                                       "/"+self.network+"_"+self.task + ".model")
                            torch.save(optimizer.state_dict(), save_dir +
                                       "/"+self.network+"_"+self.task + ".opt")

                if self.task == "tools":
                    if (results['F1-macro left'] + results['F1-macro right'])/2 >= Max_F1_macro:
                        Max_F1_macro = (
                            results['F1-macro left'] + results['F1-macro right'])/2
                        best_valid_results = results
                        if not self.debagging and not self.hyper_parameter_tuning:
                            torch.save(self.model.state_dict(), save_dir +
                                       "/"+self.network+"_"+self.task + ".model")
                            torch.save(optimizer.state_dict(), save_dir +
                                       "/"+self.network+"_"+self.task + ".opt")

                if self.task == "multi-taks":
                    if (results['F1-macro gesture'] + results['F1-macro left'] + results['F1-macro right'])/3 >= Max_F1_macro:
                        Max_F1_macro = (
                            results['F1-macro gesture'] + results['F1-macro left'] + results['F1-macro right'])/3
                        best_valid_results = results
                        if not self.debagging and not self.hyper_parameter_tuning:
                            torch.save(self.model.state_dict(), save_dir +
                                       "/"+self.network+"_"+self.task + ".model")
                            torch.save(optimizer.state_dict(), save_dir +
                                       "/"+self.network+"_"+self.task + ".opt")

                if args.upload is True:
                    wandb.log(results)

        # test HERE!!
        if self.hyper_parameter_tuning:
            return best_valid_results, eval_results_list, train_results_list, []
        else:
            best_epoch = best_valid_results['epoch']
            print(colored("model testing based on epoch: " +
                  str(best_epoch), 'green', attrs=['bold']))

            self.model.load_state_dict(torch.load(
                save_dir + "/"+self.network+"_"+self.task + ".model"))
            test_results = self.evaluate(eval_dict, batch_gen, True)
            test_results["best_epch"] = [best_epoch] * \
                len(test_results['list_of_seq'])
            return best_valid_results, eval_results_list, train_results_list, test_results

    def evaluate(self, eval_dict, batch_gen, is_test=False):
        results = {}
        device = eval_dict["device"]
        features_path = eval_dict["features_path"]
        sample_rate = eval_dict["sample_rate"]
        actions_dict = eval_dict["actions_dict_tools"]
        actions_dict_gesures = eval_dict["actions_dict_gestures"]
        ground_truth_path_right = eval_dict["gt_path_tools_right"]
        ground_truth_path_left = eval_dict["gt_path_tools_left"]
        ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            if not is_test:
                list_of_vids = batch_gen.list_of_valid_examples
            else:
                list_of_vids = batch_gen.list_of_test_examples

            recognition1_list = []
            recognition2_list = []
            recognition3_list = []

            for seq in list_of_vids:
                # print vid
                features = np.load(os.path.join(
                    features_path, seq.split('.')[0] + '.npy'))
                if batch_gen.normalization == "Min-max":
                    numerator = features.T - batch_gen.min
                    denominator = batch_gen.max - batch_gen.min
                    features = (numerator / denominator).T
                elif batch_gen.normalization == "Standard":
                    numerator = features.T - batch_gen.mean
                    denominator = batch_gen.std
                    features = (numerator / denominator).T
                elif batch_gen.normalization == "samplewise_SD":
                    samplewise_meam = features.mean(axis=1)
                    samplewise_std = features.std(axis=1)
                    numerator = features.T - samplewise_meam
                    denominator = samplewise_std
                    features = (numerator / denominator).T

                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                if self.task == "multi-taks":
                    if self.network == "LSTM" or self.network == "GRU":
                        predictions1, predictions2, predictions3 = self.model(
                            input_x, torch.tensor([features.shape[1]]))
                        predictions1 = predictions1
                        predictions1 = torch.nn.Softmax(dim=2)(predictions1)
                        predictions2 = predictions2
                        predictions2 = torch.nn.Softmax(dim=2)(predictions2)
                        predictions3 = predictions3
                        predictions3 = torch.nn.Softmax(dim=2)(predictions3)

                    else:
                        predictions1, predictions2, predictions3 = self.model(
                            input_x, torch.tensor([features.shape[1]]))
                elif self.task == "tools":
                    if self.network == "LSTM" or self.network == "GRU":
                        predictions2, predictions3 = self.model(
                            input_x, torch.tensor([features.shape[1]]))
                        predictions2 = predictions2.unsqueeze_(0)
                        predictions2 = torch.nn.Softmax(dim=2)(predictions2)
                        predictions3 = predictions3.unsqueeze_(0)
                        predictions3 = torch.nn.Softmax(dim=2)(predictions3)

                    else:
                        predictions2, predictions3 = self.model(
                            input_x, torch.tensor([features.shape[1]]))
                else:
                    if self.network == "LSTM" or self.network == "GRU":
                        predictions1 = self.model(
                            input_x, torch.tensor([features.shape[1]]))
                        predictions1 = predictions1[0].unsqueeze_(0)
                        predictions1 = torch.nn.Softmax(dim=2)(predictions1)
                    else:
                        predictions1 = self.model(
                            input_x, torch.tensor([features.shape[1]]))[0]

                if self.task == "multi-taks" or self.task == "gestures":
                    _, predicted1 = torch.max(predictions1[-1].data, 1)
                    predicted1 = predicted1.squeeze()

                if self.task == "multi-taks" or self.task == "tools":
                    _, predicted2 = torch.max(predictions2[-1].data, 1)
                    _, predicted3 = torch.max(predictions3[-1].data, 1)
                    predicted2 = predicted2.squeeze()
                    predicted3 = predicted3.squeeze()

                recognition1 = []
                recognition2 = []
                recognition3 = []
                if self.task == "multi-taks" or self.task == "gestures":
                    for i in range(len(predicted1)):
                        recognition1 = np.concatenate((recognition1, [list(actions_dict_gesures.keys())[
                            list(actions_dict_gesures.values()).index(
                                predicted1[i].item())]] * sample_rate))
                    recognition1_list.append(recognition1)
                if self.task == "multi-taks" or self.task == "tools":

                    for i in range(len(predicted2)):
                        recognition2 = np.concatenate((recognition2, [list(actions_dict.keys())[
                            list(actions_dict.values()).index(
                                predicted2[i].item())]] * sample_rate))
                    recognition2_list.append(recognition2)

                    for i in range(len(predicted3)):
                        recognition3 = np.concatenate((recognition3, [list(actions_dict.keys())[
                            list(actions_dict.values()).index(
                                predicted3[i].item())]] * sample_rate))
                    recognition3_list.append(recognition3)
            if self.task == "multi-taks" or self.task == "gestures":

                print("gestures results")
                results1, _ = metric_calculation(ground_truth_path=ground_truth_path_gestures,
                                                 recognition_list=recognition1_list, list_of_videos=list_of_vids,
                                                 suffix="gesture", is_test=is_test)

                results.update(results1)

            if self.task == "multi-taks" or self.task == "tools":

                print("right hand results")
                results2, _ = metric_calculation(ground_truth_path=ground_truth_path_right,
                                                 recognition_list=recognition2_list, list_of_videos=list_of_vids,
                                                 suffix="right", is_test=is_test)
                print("left hand results")
                results3, _ = metric_calculation(ground_truth_path=ground_truth_path_left,
                                                 recognition_list=recognition3_list, list_of_videos=list_of_vids,
                                                 suffix="left", is_test=is_test)
                results.update(results2)
                results.update(results3)

            if is_test:
                results["list_of_seq"] = list_of_vids
            self.model.train()
            return results
