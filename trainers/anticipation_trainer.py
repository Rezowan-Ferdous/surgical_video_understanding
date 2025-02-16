import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import numpy as np
from utils.anticipationutils import cal_performance, normalize_duration

#FUTR training function
def train(args, model, train_loader, optimizer, scheduler, criterion,  model_save_path, pad_idx, device):
    model.to(device)
    model.train()
    print("Training Start")
    for epoch in range(args.epochs):
        epoch_acc =0
        epoch_loss = 0
        epoch_loss_class = 0
        epoch_loss_dur = 0
        epoch_loss_seg = 0
        total_class = 0
        total_class_correct = 0
        total_seg = 0
        total_seg_correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            features, past_label, trans_dur_future, trans_future_target = data
            print(f"Features Shape: {features.shape}, Past Label Shape: {past_label.shape}")
            features = features.to(device) #[B, S, C]
            past_label = past_label.to(device) #[B, S]
            trans_dur_future = trans_dur_future.to(device)
            trans_future_target = trans_future_target.to(device)
            trans_dur_future_mask = (trans_dur_future != pad_idx).long().to(device)

            B = trans_dur_future.size(0)
            target_dur = trans_dur_future*trans_dur_future_mask
            target = trans_future_target
            if args.input_type == 'i3d_transcript':
                inputs = (features, past_label)
            elif args.input_type == 'gt':
                gt_features = past_label.int()
                inputs = (gt_features, past_label)

            outputs = model(inputs)
            print(f"Outputs: {outputs}")
            losses = 0
            if args.seg :
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                output_seg = output_seg.view(-1, C).to(device)
                target_past_label = past_label.view(-1)
                loss_seg, n_seg_correct, n_seg_total = cal_performance(output_seg, target_past_label, pad_idx)
                losses += loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                epoch_loss_seg += loss_seg.item()
            if args.anticipate :
                output = outputs['action']
                B, T, C = output.size()
                output = output.view(-1, C).to(device)
                target = target.contiguous().view(-1)
                out = output.max(1)[1] #oneshot
                out = out.view(B, -1)
                loss, n_correct, n_total = cal_performance(output, target, pad_idx)
                acc = n_correct / n_total
                loss_class = loss.item()
                losses += loss
                total_class += n_total
                total_class_correct += n_correct
                epoch_loss_class += loss_class

                output_dur = outputs['duration']
                output_dur = normalize_duration(output_dur, trans_dur_future_mask)
                target_dur = target_dur * trans_dur_future_mask
                loss_dur = torch.sum(criterion(output_dur, target_dur)) / \
                torch.sum(trans_dur_future_mask)

                losses += loss_dur
                epoch_loss_dur += loss_dur.item()


            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()


        epoch_loss = epoch_loss / (i+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss)
        if args.anticipate :
            accuracy = total_class_correct/total_class
            epoch_loss_class = epoch_loss_class / (i+1)
            print('Training Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )
            if args.task == 'long' :
                epoch_loss_dur = epoch_loss_dur / (i+1)
                print('dur loss: %.5f'%epoch_loss_dur)

        if args.seg :
            acc_seg = total_seg_correct / total_seg
            epoch_loss_seg = epoch_loss_seg / (i+1)
            print('seg loss :%.3f'%epoch_loss_seg, ', seg acc : %.5f'%acc_seg)

        scheduler.step()

        save_path = os.path.join(model_save_path)
        if epoch >= 30 :
            save_file = os.path.join(save_path, 'checkpoint'+str(epoch)+'.ckpt')
            torch.save(model.state_dict(), save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    return model


import torch
import torch.nn as nn
import numpy
import pdb
import os
import copy
from collections import defaultdict
import numpy as np
from utils.anticipationutils import normalize_duration, eval_file

#FUTR predict function
def predict(model, vid_list, args, obs_p, n_class, actions_dict, device):
    model.eval()
    with torch.no_grad():
        data_path = './datasets'
        if args.dataset == 'breakfast':
            data_path = os.path.join(data_path, 'breakfast')
        elif args.dataset == '50salads':
            data_path = os.path.join(data_path, '50salads')
        gt_path = os.path.join(data_path, 'groundTruth')
        features_path = os.path.join(data_path, 'features')

        eval_p = [0.1, 0.2, 0.3, 0.5]
        pred_p = 0.5
        sample_rate = args.sample_rate
        NONE = n_class-1
        T_actions = np.zeros((len(eval_p), len(actions_dict)))
        F_actions = np.zeros((len(eval_p), len(actions_dict)))
        actions_dict_with_NONE = copy.deepcopy(actions_dict)
        actions_dict_with_NONE['NONE'] = NONE

        for vid in vid_list:
            file_name = vid.split('/')[-1].split('.')[0]

            # load ground truth actions
            gt_file = os.path.join(gt_path, file_name+'.txt')
            gt_read = open(gt_file, 'r')
            gt_seq = gt_read.read().split('\n')[:-1]
            gt_read.close()

            # load features
            features_file = os.path.join(features_path, file_name+'.npy')
            features = np.load(features_file).transpose()

            vid_len = len(gt_seq)
            past_len = int(obs_p*vid_len)
            future_len = int(pred_p*vid_len)

            past_seq = gt_seq[:past_len]
            features = features[:past_len]
            inputs = features[::sample_rate, :]
            inputs = torch.Tensor(inputs).to(device)

            outputs = model(inputs=inputs.unsqueeze(0), mode='test')
            output_action = outputs['action']
            output_dur = outputs['duration']
            output_label = output_action.max(-1)[1]

            # fine the forst none class
            none_mask = None
            for i in range(output_label.size(1)) :
                if output_label[0,i] == NONE :
                    none_idx = i
                    break
                else :
                    none = None
            if none_idx is not None :
                none_mask = torch.ones(output_label.shape).type(torch.bool)
                none_mask[0, none_idx:] = False

            output_dur = normalize_duration(output_dur, none_mask.to(device))

            pred_len = (0.5+future_len*output_dur).squeeze(-1).long()

            pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
            predicted = torch.ones(future_len)
            action = output_label.squeeze()

            for i in range(len(action)) :
                predicted[int(pred_len[i]) : int(pred_len[i] + pred_len[i+1])] = action[i]
                pred_len[i+1] = pred_len[i] + pred_len[i+1]
                if i == len(action) - 1 :
                    predicted[int(pred_len[i]):] = action[i]


            prediction = past_seq
            for i in range(len(predicted)):
                prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]]))

            #evaluation
            for i in range(len(eval_p)):
                p = eval_p[i]
                eval_len = int((obs_p+p)*vid_len)
                eval_prediction = prediction[:eval_len]
                T_action, F_action = eval_file(gt_seq, eval_prediction, obs_p, actions_dict)
                T_actions[i] += T_action
                F_actions[i] += F_action

        results = []
        for i in range(len(eval_p)):
            acc = 0
            n = 0
            for j in range(len(actions_dict)):
                total_actions = T_actions + F_actions
                if total_actions[i,j] != 0:
                    acc += float(T_actions[i,j]/total_actions[i,j])
                    n+=1

            result = 'obs. %d '%int(100*obs_p) + 'pred. %d '%int(100*eval_p[i])+'--> MoC: %.4f'%(float(acc)/n)
            results.append(result)
            print(result)
        print('--------------------------------')

        return





