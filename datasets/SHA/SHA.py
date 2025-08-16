## 
import os
import random
from scipy import spatial
import networkx as nx

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = os.path.join(self.root_path, "train.txt")
        self.eval_list = os.path.join(self.root_path, "test.txt")
        # 
        if train:
            self.img_list_file = [name.split(',') for name in open(self.train_lists).read().splitlines()]
        else:
            self.img_list_file = [name.split(',') for name in open(self.eval_list).read().splitlines()]

        self.img_list = self.img_list_file
        
        # 
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index][0]
        # print("hello", img_path)
        gt_path = self.img_list[index][1]
        # print("hello gt", gt_path)
        # 
        img, point = load_data((img_path, gt_path), self.train)
        #
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.5, 1.4]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 224:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        # random crop augumentaiton
        if self.train and self.patch:
            img, point = random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        if random.random() > 0.1 and self.train and self.flip: # never flip
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 224 - point[i][:, 0]
        # random change brightness
        if random.random() > 0.3 and self.train: # never flip
            #
            img = (torch.Tensor(img).clone())*random.uniform(8,12)/10
            for i, _ in enumerate(point):
                point[i][:, 0] = point[i][:, 0]

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        #  need to adapt your own image names
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            #image_id_1 = int(img_path.split('/')[-1].split('.')[0].split("_")[1][4:8])
            image_id_1 = int(img_path.split('/')[-1].split('.')[0].split("_")[1])
            image_id_1 = torch.Tensor([image_id_1]).long()
            #
            image_id_2 = int(img_path.split('/')[-1].split('.')[0].split("_")[1])
            image_id_2 = torch.Tensor([image_id_2]).long()
            target[i]['image_id_1'] = image_id_1
            target[i]['image_id_2'] = image_id_2
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()
            

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    #
    pts = open(gt_path).read().splitlines()
    for pt_0 in pts:
        #pt = eval(pt_0) 
        pt = pt_0.split(" ")
        x = float(pt[0])
        y = float(pt[1])
        points.append([x, y])
    return img, np.array(points)

# random crop augumentation
def random_crop(img, den, num_patch=4):
    half_h = 224
    half_w = 224
    
    #half_h = 128
    #half_w = 128
    
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # 
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # 
        result_img[i] = img[:, start_h:end_h, start_w:end_w]#*random.uniform(5,15)/10
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # 
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den


# Define the MosaicAugmentation class that was referenced in the augmentations
# class SHA(Dataset):
#     def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
#         self.root_path = data_root
#         self.train_lists = os.path.join(self.root_path, "train.txt")
#         self.eval_list = os.path.join(self.root_path, "test.txt")
        
#         if train:
#             self.img_list_file = [name.split(',') for name in open(self.train_lists).read().splitlines()]
#         else:
#             self.img_list_file = [name.split(',') for name in open(self.eval_list).read().splitlines()]

#         self.img_list = self.img_list_file
        
#         self.nSamples = len(self.img_list)
        
#         self.transform = transform
#         self.train = train
#         self.patch = patch
#         self.flip = flip
        
#         # Initialize the augmentations with standard Albumentations transforms
#         self.augmentations = A.Compose([
#             A.OneOf([
#                 A.GaussianBlur(blur_limit=(3, 7), p=0.5),
#                 A.MotionBlur(blur_limit=(3, 7), p=0.5),
#             ], p=0.5),
#             A.OneOf([
#                 A.GaussNoise(var_limit=(10, 50), p=0.5),
#                 A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
#             ], p=0.5),
#             A.OneOf([
#                 A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#                 A.RandomGamma(gamma_limit=(80, 120), p=0.5),
#             ], p=0.5),
#             # Removed custom MosaicAugmentation
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

#     def __len__(self):
#         return self.nSamples

#     def __getitem__(self, index):
#         assert index <= len(self), 'index range error'

#         img_path = self.img_list[index][0]
#         gt_path = self.img_list[index][1]
        
#         img, point = load_data((img_path, gt_path), self.train)
        
#         # Convert PIL Image to numpy array for processing
#         img_np = np.array(img)
        
#         # Apply the augmentations if in training mode
#         if self.train:
#             # Apply Albumentations transforms
#             # Make sure point coordinates are in the correct format for Albumentations
#             keypoints = [(float(p[0]), float(p[1])) for p in point]
            
#             try:
#                 # Apply augmentations
#                 augmented = self.augmentations(image=img_np, keypoints=keypoints)
#                 img_np = augmented['image']
#                 point = np.array(augmented['keypoints']) if augmented['keypoints'] else point
#             except Exception as e:
#                 print(f"Augmentation error: {e}")
#                 # If augmentation fails, continue with original image
#                 pass
        
#         # Convert back to PIL Image for the PyTorch transforms
#         img = Image.fromarray(img_np.astype('uint8') if img_np.dtype != np.uint8 else img_np)
        
#         # Apply additional PyTorch transforms if provided
#         if self.transform is not None:
#             img = self.transform(img)

#         if self.train:
#             # data augmentation -> random scale
#             scale_range = [0.5, 1.4]
#             min_size = min(img.shape[1:])
#             scale = random.uniform(*scale_range)
#             # scale the image and points
#             if scale * min_size > 224:
#                 img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
#                 point *= scale
                
#         # random crop augumentaiton
#         if self.train and self.patch:
#             img, point = random_crop(img, point)
#             for i, _ in enumerate(point):
#                 point[i] = torch.Tensor(point[i])
                
#         # random flipping (now handled by albumentations but kept for compatibility)
#         if random.random() > 0.1 and self.train and self.flip:
#             # random flip
#             img = torch.Tensor(img[:, :, :, ::-1].copy())
#             for i, _ in enumerate(point):
#                 point[i][:, 0] = 224 - point[i][:, 0]
                
#         # random change brightness (now handled by albumentations but kept for compatibility)
#         if random.random() > 0.3 and self.train:
#             img = (torch.Tensor(img).clone())*random.uniform(8,12)/10
#             for i, _ in enumerate(point):
#                 point[i][:, 0] = point[i][:, 0]

#         if not self.train:
#             point = [point]

#         img = torch.Tensor(img)
        
#         # Prepare target dictionary
#         target = [{} for i in range(len(point))]
#         for i, _ in enumerate(point):
#             target[i]['point'] = torch.Tensor(point[i])
#             image_id_1 = int(img_path.split('/')[-1].split('.')[0].split("_")[1])
#             image_id_1 = torch.Tensor([image_id_1]).long()
#             image_id_2 = int(img_path.split('/')[-1].split('.')[0].split("_")[1])
#             image_id_2 = torch.Tensor([image_id_2]).long()
#             target[i]['image_id_1'] = image_id_1
#             target[i]['image_id_2'] = image_id_2
#             target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

#         return img, target


# def load_data(img_gt_path, train):
#     img_path, gt_path = img_gt_path
#     # load the images
#     img = cv2.imread(img_path)
#     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # load ground truth points
#     points = []
#     #
#     pts = open(gt_path).read().splitlines()
#     for pt_0 in pts:
#         pt = pt_0.split(" ")
#         x = float(pt[0])
#         y = float(pt[1])
#         points.append([x, y])
#     return img, np.array(points)

# # random crop augumentation
# def random_crop(img, den, num_patch=4):
#     half_h = 224
#     half_w = 224
    
#     result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
#     result_den = []
    
#     for i in range(num_patch):
#         start_h = random.randint(0, img.size(1) - half_h)
#         start_w = random.randint(0, img.size(2) - half_w)
#         end_h = start_h + half_h
#         end_w = start_w + half_w
        
#         result_img[i] = img[:, start_h:end_h, start_w:end_w]
#         # copy the cropped points
#         idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        
#         record_den = den[idx]
#         record_den[:, 0] -= start_w
#         record_den[:, 1] -= start_h

#         result_den.append(record_den)

#     return result_img, result_den