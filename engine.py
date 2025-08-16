# 
import math
import os
import sys
from typing import Iterable
import torch
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2
from scipy import spatial
import networkx as nx

from matplotlib import pyplot as plt



import util.misc as utils
#from util.misc import NestedTensor, MetricLogger


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
#
def vis(samples, targets, pred, vis_dir, epoch, predict_cnt, gt_cnt):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # 
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, :].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, :].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 5
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name_1 = targets[idx]['image_id_1']
        name_2 = targets[idx]['image_id_2']
        #################
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(sample_gt)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(sample_pred)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        fig.suptitle('manual count=%4.2f, inferred count=%4.2f'%(gt_cnt, predict_cnt), fontsize=10)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95]) # maize tassels counting
        plt.savefig(os.path.join(vis_dir, '{}_{}_id_{}_ind_{}.jpg'.format(epoch, idx, int(name_1), int(name_2))), bbox_inches='tight', dpi = 300)
        plt.close()

def load_checkpoint(model, optimizer, filename='/home/tejasri/Pardhu/P2PNew/Jupyter-UI/ckpt_p2pnetsoy/latest.pth'):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return model, optimizer, start_epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return model, optimizer, 0, None
    

# the training
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    for samples, targets in data_loader:
        #
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        #
        # calc the losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce all losses (get the mean values)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# evaluate the model performance during training
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, epoch, threshold, vis_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    for samples, targets in data_loader:

        samples = samples.to(device)

        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = threshold

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()#.tolist()
        # choose to merge closely located points
        if points.shape[0]<10000 and points.shape[0] != 0:
            # choose the cut off point
            cutoff = 500/points.shape[0]
            if cutoff<20:
                cutoff = 20
            components = nx.connected_components(
                nx.from_edgelist(
                    (i, j) for i, js in enumerate(
                        spatial.KDTree(points).query_ball_point(points, cutoff)
                    )
                    for j in js
                )
            )

            clusters = {j: i for i, js in enumerate(components) for j in js}

            # reorganize the points to the order of clusters 
            points_reo = np.zeros(points.shape)
            i = 0
            for key in clusters.keys():
                points_reo[i,:] = points[key,:]
                i+=1
            # points_n has the same order as clusters
            res = [clusters[key] for key in clusters.keys()]
            res_n = np.array(res).reshape(-1,1)

            points_n = []
            for i in np.unique(res_n):
                tmp = points_reo[np.where(res_n[:,0] == i)]
                points_n.append( [np.mean(tmp[:,0]), np.mean(tmp[:,1])])
        else:
            points_n = points.tolist()

        predict_cnt = len(points_n)
        #save the visualized images
        if vis_dir is not None: 
            vis(samples, targets, [points_n], vis_dir, epoch, predict_cnt, gt_cnt)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse
