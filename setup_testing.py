import argparse
import datetime
import random
import time
from pathlib import Path
#import tent

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

args = {}

args["backbone"] = "efficientnet-b0"
args["row"] = 2
args["line"] = 2
args["weight_path"] = '/u/student/2023/ai23mtech02003/yuvraj/Pardhu/paddy_exp/ckpt_efficientnet_focal_hungarian/best_mae.pth'
args["vis_dir"] = '/u/student/2023/ai23mtech02003/yuvraj/Pardhu/paddy_exp/new_results/vis_efficientnet_focal'

#os.makedirs(args["vis_dir"], exist_ok=True)

def read_gt_points(gt_txt_path):
    """Read ground truth points from txt file."""
    points = []
    if not os.path.exists(gt_txt_path):
        print(f"Warning: Ground truth file not found: {gt_txt_path}")
        return np.array([])
        
    try:
        with open(gt_txt_path, 'r') as f:
            lines = f.readlines()
            print(f"Found {len(lines)} points in {gt_txt_path}")  # Debug print
            for line in lines:
                try:
                    # Split by any whitespace
                    coords = line.strip().split()
                    # Extract x, y coordinates
                    x = float(coords[0])
                    y = float(coords[1])
                    points.append([x, y])
                except Exception as e:
                    print(f"Error parsing line in {gt_txt_path}: {line.strip()}")
                    continue
    except Exception as e:
        print(f"Error reading ground truth file {gt_txt_path}: {e}")
        return np.array([])
    
    return np.array(points)

def draw_x_mark(img, center, size, color, thickness):
    """Draw X mark on the image."""
    px, py = int(center[0]), int(center[1])
    cv2.line(img, (px - size, py - size), (px + size, py + size), color, thickness)
    cv2.line(img, (px + size, py - size), (px - size, py + size), color, thickness)
    return img

def visualize_points(img, pred_points, gt_points, alpha=0.5, mark_size=4, thickness=2):
    """Draw predictions as X marks and ground truth as dots."""
    img_overlay = img.copy()
    
    # Draw ground truth points as green dots
    for p in gt_points:
        px, py = int(round(p[0])), int(round(p[1]))
        cv2.circle(img_overlay, (px, py), mark_size, (0, 255, 0), -1)
    
    # Draw prediction points as red X marks
    for p in pred_points:
        if isinstance(p, (list, tuple)):
            px, py = int(round(p[0])), int(round(p[1]))
        else:
            px, py = int(round(p[0])), int(round(p[1]))
        draw_x_mark(img_overlay, (px, py), mark_size, (0, 0, 255), thickness)
    
    return cv2.addWeighted(img_overlay, alpha, img, 1 - alpha, 0)

def main(args):
    device = torch.device('cuda')
    model = build_model(args)
    model.to(device)
    if args["weight_path"] is not None:
        checkpoint = torch.load(args["weight_path"], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()
    
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    #set the dataset
    img_list_path = '/u/student/2023/ai23mtech02003/yuvraj/Pardhu/P2PNet-paddy/Paddy_dataset/test.txt'
    img_list = [name.split(',') for name in open(img_list_path).read().splitlines()]

    img_type = "*.jpg"

    csv_name = (args["vis_dir"]).split("/")[-1]
    loss_csv = open(os.path.join(args["vis_dir"], '{}_seed_counting_{}.csv'.format(csv_name, img_type)), 'w+')

    for img_path_ij, _ in img_list:
        img_name = os.path.basename(img_path_ij)
        img_name_pred_before = img_name.replace(".jpg", '_pred_bf.jpg')
        img_name_pred_after = img_name.replace(".jpg", '_pred_af.jpg')
        
        img_0 = cv2.imread(img_path_ij)
        img_raw = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
        img = transform(img_raw)
        height, width = img_0.shape[:2]

        # Load ground truth count from the corresponding .txt file
        gt_txt_path = os.path.join("/u/student/2023/ai23mtech02003/yuvraj/Pardhu/P2PNet-paddy/Paddy_dataset/test/labels", os.path.splitext(img_name)[0] + ".txt")
        gt_points = read_gt_points(gt_txt_path)
        gt_count = len(gt_points)
        # with open(gt_txt_path, 'r') as f:
        #     gt_count = len(f.readlines())
        #     #print(gt_count)

        
        new_width = (width // 128 +1) * 128
        new_height = (height // 128 + 1)* 128
        img_in = torch.zeros((3, new_height, new_width))
        img_in[:, :height, :width] = img
        samples = img_in.unsqueeze(0).to(device)
        

        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()#.tolist()
        print("after filtering", len(points))
        predict_cnt = int((outputs_scores > threshold).sum())
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        # draw the predictions
        size = 6
        alpha = 0.7
        img_to_draw_before = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        img_to_draw_before_in_x = img_to_draw_before.copy()
        
        # Draw ground truth points in green
        for p in gt_points:
            try:
                px, py = int(round(p[0])), int(round(p[1]))
                if 0 <= px < width and 0 <= py < height:
                    # Draw green filled circle
                    cv2.circle(img_to_draw_before_in_x, (px, py), 5, (0, 255, 0), -1)
            except Exception as e:
                print(f"Error drawing GT point {p}: {e}")

        for p in points:
            try:
                px, py = int(round(p[0])), int(round(p[1]))
                if 0 <= px < width and 0 <= py < height:
                    size = 4  # Size of the X mark
                    thickness = 2  # Thickness of the lines
                    # Draw the two lines of the X
                    cv2.line(img_to_draw_before_in_x, 
                            (px - size, py - size), 
                            (px + size, py + size), 
                            (0, 0, 255), 
                            thickness)
                    cv2.line(img_to_draw_before_in_x, 
                            (px + size, py - size), 
                            (px - size, py + size), 
                            (0, 0, 255), 
                            thickness)
            except Exception as e:
                print(f"Error drawing prediction point {p}: {e}")
                
        
        img_to_draw_before_in_xx = cv2.addWeighted(img_to_draw_before_in_x, alpha, img_to_draw_before, 1 - alpha, 0)
        
        #Add text annotations with background for better visibility
        def draw_text_with_background(img, text, pos, color):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 2.0 # Increased from 1.0 to 1.5
            thickness = 4  # Increased from 2 to 3
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            
            # Draw background rectangle
            padding = 15 # Slightly increased for larger font
            cv2.rectangle(img, 
                        (pos[0]-padding, pos[1]-text_h-padding), 
                        (pos[0]+text_w+padding, pos[1]+padding), 
                        (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(img, text, pos, font, scale, color, thickness)
            
        draw_text_with_background(img_to_draw_before_in_xx, f"GT Count: {gt_count}", (10, 50), (0, 255, 0))
        draw_text_with_background(img_to_draw_before_in_xx, f"Pred Count: {predict_cnt}", (10, 130), (0, 0, 255))
            
        # save the visualized image
        cv2.imwrite(os.path.join(args["vis_dir"], img_name_pred_before), img_to_draw_before_in_xx[:height,:width,:])
        
        # if points.shape[0]< 10000 and points.shape[0] != 0:
        #     #print("doing clustering")
        #     cutoff = 500/points.shape[0]
        #     if cutoff<20:
        #         cutoff = 20
        #         components = nx.connected_components(
        #             nx.from_edgelist(
        #                 (i, j) for i, js in enumerate(
        #                     spatial.KDTree(points).query_ball_point(points, cutoff))
        #                 for j in js
        #             )
        #         )
        #         clusters = {j: i for i, js in enumerate(components) for j in js}

        #         # reorganize the points to the order of clusters 
        #         points_reo = np.zeros(points.shape)
        #         i = 0
        #         for key in clusters.keys():
        #             #print(key)
        #             points_reo[i,:] = points[key,:]
        #             i+=1
        #         # points_n has the same order as clusters
        #         res = [clusters[key] for key in clusters.keys()]
        #         res_n = np.array(res).reshape(-1,1)

        #         points_n = []
        #         for i in np.unique(res_n):
        #             tmp = points_reo[np.where(res_n[:,0] == i)]
        #             points_n.append( [np.mean(tmp[:,0]), np.mean(tmp[:,1])])
        #     else:
        #         points_n = points.tolist()
        #     # calculate the distance and find the center of the too close points

        #     #predict_cnt = int((outputs_scores > threshold).sum())
        #     predict_cnt_before = len(points)
        #     predict_cnt_after = len(points_n)
        
        #     def draw_text_with_background(img, text, pos, color):
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         scale = 2.0  # Larger text size
        #         thickness = 4  # Bolder text
        #         # Get text size
        #         (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
                
        #         # Draw background rectangle
        #         padding = 15  # Increased padding for larger text
        #         cv2.rectangle(img, 
        #                     (pos[0]-padding, pos[1]-text_h-padding), 
        #                     (pos[0]+text_w+padding, pos[1]+padding), 
        #                     (0, 0, 0), -1)
                
        #         # Draw text
        #         cv2.putText(img, text, pos, font, scale, color, thickness)
                
        #     # draw the predictions
        #     alpha = 0.5
        #     #. before 
        #     size = 6
        #     #
        #     img_before = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        #     img_viz_before = visualize_points(img_before, points, gt_points)
            
        #     # Define height and width for slicing
        #     height, width = img_before.shape[:2]
            
        #     # Add text for before clustering
        #     # cv2.putText(img_viz_before, f"GT Count: {gt_count}", (10, 50), 
        #     #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        #     # cv2.putText(img_viz_before, f"Pred Count: {predict_cnt_before}", (10, 130), 
        #     #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            
        #     draw_text_with_background(img_viz_before, f"GT Count: {gt_count}", (10, 50), (0, 255, 0))
        #     draw_text_with_background(img_viz_before, f"Pred Count: {predict_cnt_before}", (10, 110), (0, 0, 255))
            
        #     # Visualize after clustering
        #     img_after = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        #     img_viz_after = visualize_points(img_after, points_n, gt_points)
            
        #     # Add text for after clustering
        #     # cv2.putText(img_viz_after, f"GT Count: {gt_count}", (10, 30), 
        #     #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     # cv2.putText(img_viz_after, f"Pred Count: {predict_cnt_after}", (10, 70), 
        #     #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        #     draw_text_with_background(img_viz_after, f"GT Count: {gt_count}", (10, 50), (0, 255, 0))
        #     draw_text_with_background(img_viz_after, f"Pred Count: {predict_cnt_after}", (10, 130), (0, 0, 255))
            
        #     # Save images
        #     img_name_pred_before = img_name.replace(".jpg", '_pred_bf.jpg')
        #     img_name_pred_after = img_name.replace(".jpg", '_pred_af.jpg')
            
        #     cv2.imwrite(os.path.join(args["vis_dir"], img_name_pred_before), 
        #             img_viz_before[:height,:width,:])
        #     cv2.imwrite(os.path.join(args["vis_dir"], img_name_pred_after), 
        #             img_viz_after[:height,:width,:])

        #     print(f"Number of seeds before = {predict_cnt_before}")
        #     print(f"Number of seeds after = {predict_cnt_after}")
            
            
        # # #save the detected pod number
        # loss_csv.write(f'{img_name_pred_before.split(".")[0]},{predict_cnt_before},'
        #                 f'{img_name_pred_after.split(".")[0]},{predict_cnt_after},{gt_count}\n')
        loss_csv.write('{},{},{}\n'.format(img_name_pred_before.split(".")[0], predict_cnt, gt_count))
        loss_csv.flush()  
    loss_csv.close


if __name__ == '__main__':
    main(args)
