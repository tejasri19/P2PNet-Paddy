import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                    accuracy, get_world_size, interpolate,
                    is_dist_avail_and_initialized)
from torchvision import models
import numpy as np
from models.matcher import build_matcher_crowd

class SimAM(nn.Module):
    """Simplified Attention Module"""
    def __init__(self, channels):
        super(SimAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.e = 1e-4

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Calculate spatial attention
        n = w * h - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        d_norm = d / (4 * (v + self.e))
        
        attention = torch.sigmoid(-d_norm)
        return x * self.gamma * attention + x

class SoftPool(nn.Module):
    """Soft Pooling module"""
    def __init__(self):
        super(SoftPool, self).__init__()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1)
        weights = self.softmax(x_flat)
        pooled = (x_flat * weights).sum(dim=2)
        return pooled.view(b, c, 1, 1)

class MSFF(nn.Module):
    """Multi-Scale Feature Fusion module"""
    def __init__(self, c1, c2, c3, out_channels):
        super(MSFF, self).__init__()
        self.conv1 = nn.Conv2d(c1, out_channels, 1)
        self.conv2 = nn.Conv2d(c2, out_channels, 1)
        self.conv3 = nn.Conv2d(c3, out_channels, 1)
        
        self.softpool = SoftPool()
        self.simam = SimAM(out_channels)
        
        # Upsampling layers
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
    def forward(self, f1, f2, f3):
        # Process each feature level
        f1 = self.conv1(f1)  # H x W
        f2 = self.conv2(f2)  # H/2 x W/2
        f3 = self.conv3(f3)  # H/4 x W/4
        
        # Upsample f2 and f3 to match f1's size
        f2_up = self.upsample2(f2)
        f3_up = self.upsample4(f3)
        
        # Apply SoftPool to f1 and f2
        p1 = self.softpool(f1)
        p2 = self.softpool(f2)
        
        # Expand pooled features to match spatial dimensions
        p1 = p1.expand(-1, -1, f1.size(2), f1.size(3))
        p2 = p2.expand(-1, -1, f1.size(2), f1.size(3))
        
        # Fusion
        fused = f3_up + f2_up + f1
        
        # Apply SimAM
        out = self.simam(fused)
        return out

class MLCA(nn.Module):
    def __init__(self, in_channels):
        super(MLCA, self).__init__()
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_att = self.spatial_pool(x)
        channel_att = self.channel_conv(channel_att)
        channel_att = self.sigmoid(channel_att)
        
        spatial_att = self.spatial_conv(x)
        spatial_att = self.sigmoid(spatial_att)
        
        return x * channel_att * spatial_att

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.mlca = MLCA(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.mlca(out)
        out += self.shortcut(x)
        return F.relu(out)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchor_points = num_anchor_points
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        
        # Output layer - directly predict coordinates for each anchor point
        #self.output_conv = nn.Conv2d(feature_size, 2 * self.num_anchor_points, kernel_size=1)
        
        # Changed: separate regression head for each anchor point
        self.regression_heads = nn.ModuleList([
            nn.Conv2d(feature_size, 2, kernel_size=1)  # 2 for x,y coordinates
            for _ in range(num_anchor_points)
        ])
        
    def forward(self, x):
        
        h, w = x.shape[2:]
        scale_factor = torch.tensor([w, h], device=x.device)

        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        # Output layer
        # x = self.output_conv(x)
        
        # # Global average pooling
        # x = x.mean(dim=(2, 3))  # Average over spatial dimensions
        
        # # Reshape to [batch_size, num_anchor_points, 2]
        # batch_size = x.size(0)
        # x = x.view(batch_size, self.num_anchor_points, 2)
        
        # Get predictions for each anchor point
        batch_size = x.shape[0]
        predictions = []
        for head in self.regression_heads:
            # Get prediction for this anchor point
            pred = head(x)
            # Global average pooling
            pred = pred.mean(dim=(2, 3))  # [batch_size, 2]
            predictions.append(pred)
            
        # Stack predictions
        x = torch.stack(predictions, dim=1)  # [batch_size, num_anchor_points, 2]
        
        x = x / scale_factor  # Normalize coordinates
        
        return x

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=1, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        
        # Output layer - 
        #self.output_conv = nn.Conv2d(feature_size, 2 * self.num_anchor_points, kernel_size=1)
        self.classification_heads = nn.ModuleList([
            nn.Conv2d(feature_size, 2, kernel_size=1)  # 2 for binary classification
            for _ in range(num_anchor_points)
        ])
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        # # Output layer
        # x = self.output_conv(x)
        
        # # Global average pooling
        # x = x.mean(dim=(2, 3))  # Average over spatial dimensions
        
        
        # batch_size = x.size(0)
        # x = x.view(batch_size, self.num_anchor_points, 2)
        
        # Get predictions for each anchor point
        batch_size = x.shape[0]
        predictions = []
        for head in self.classification_heads:
            # Get prediction for this anchor point
            pred = head(x)
            # Global average pooling
            pred = pred.mean(dim=(2, 3))  # [batch_size, 2]
            predictions.append(pred)
            
        # Stack predictions
        x = torch.stack(predictions, dim=1)  # [batch_size, num_anchor_points, 2]
        
        x = F.softmax(x, dim=-1)
        return x
    
    
def generate_anchor_points(stride=16, row=2, line=2):
    row_step = stride / row
    line_step = stride / line
    
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    
    return anchor_points

# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=2, line=2):
        super(AnchorPoints, self).__init__()
        # Use only one pyramid level since we want exactly 4 points
        self.pyramid_levels = [3] if pyramid_levels is None else pyramid_levels
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.row = row
        self.line = line
        self.num_points = row * line  # Should be 4 (2x2)

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        
        # Generate exactly 4 anchor points
        anchor_points = generate_anchor_points(
            stride=2**self.pyramid_levels[0], 
            row=self.row, 
            line=self.line
        )
        
        # Get shifted points for the first (and only) pyramid level
        shifted_anchor_points = shift(image_shapes[0], self.strides[0], anchor_points)
        
        # Take exactly 4 points
        if len(shifted_anchor_points) > 4:
            shifted_anchor_points = shifted_anchor_points[:4]
            
        # Expand dimensions for batch size
        all_anchor_points = np.expand_dims(shifted_anchor_points, axis=0)
        
        return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda() if torch.cuda.is_available() else torch.from_numpy(all_anchor_points.astype(np.float32))


class RiceSeedlingDetector(nn.Module):
    def __init__(self, backbone=None, row=2, line=2, hidden_dim=256):
        super(RiceSeedlingDetector, self).__init__()
        
        # Load pretrained ResNet34
        self.backbone = models.resnet34(pretrained=True)
        
        # Remove the last fully connected layer and avgpool
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # ResNet34 output channels
        backbone_out_channels = 512  # ResNet34's last conv layer has 512 channels
        
        # MSFF neck for feature fusion
        self.msff = MSFF(128, 256, backbone_out_channels, hidden_dim*2)  # Adjusted for ResNet34 channels
        
        # Number of anchor points
        self.num_anchor_points = row * line
        
        
        # Detection heads
        self.regression = RegressionModel(hidden_dim*2, num_anchor_points=self.num_anchor_points, feature_size=hidden_dim)
        self.classification = ClassificationModel(hidden_dim*2, num_anchor_points=self.num_anchor_points, num_classes=1, feature_size=hidden_dim)
        
        # Anchor points generator
        self.anchor_points = AnchorPoints(pyramid_levels=[3,], row=row, line=line)
        
        # Freeze early layers of ResNet
        self._freeze_backbone_layers()

    def _freeze_backbone_layers(self):
        # Freeze all BatchNorm layers
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        
        # Freeze first two layers
        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

    def forward(self, x):
        if isinstance(x, NestedTensor):
            x = x.tensors
            
        # Get intermediate features from ResNet34
        features = []
        for i, block in enumerate(self.backbone.children()):
            x = block(x)
            if i in [5, 6, 7]:  # After layer2, layer3, layer4
                features.append(x)
        
        f1, f2, f3 = features  # ResNet34 features from different stages
        
        # Multi-scale feature fusion
        fused_features = self.msff(f1, f2, f3)
        
        # Detection heads
        regression = self.regression(fused_features) * 100 # Scale factor for better gradient flow
        classification = self.classification(fused_features)
        
        # Generate anchor points
        batch_size = x.shape[0]
        anchor_points = self.anchor_points(x)
        
        # Ensure anchor_points has correct batch dimension
        if anchor_points.size(0) == 1:
            anchor_points = anchor_points.repeat(batch_size, 1, 1)
            
        #print("Debug shapes in forward pass:")
        #print(f"Classification shape: {classification.shape}")
        #print(f"Regression shape: {regression.shape}")
        #print(f"Anchor points shape: {anchor_points.shape}")
        
        
        
        # Output predictions
        output_coord = regression + anchor_points
        output_class = classification
        
        
        return {'pred_logits': output_class, 'pred_points': output_coord}
        
        
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        #self.density_alpha = density_alpha
        
        # Adding +1 for background class
        empty_weight = torch.ones(2)  # [background, foreground]
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], 0,
        #                             dtype=torch.int64, device=src_logits.device)
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64,
                                device=src_logits.device)
        target_classes[idx] = 1
        #target_classes = torch.zeros_like(src_logits, dtype=torch.float32)

        # Use binary cross entropy loss
        loss_ce = F.cross_entropy(src_logits.view(-1, 2), target_classes.view(-1), self.empty_weight)
        losses = {'loss_ce': loss_ce}
        
        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Normalize points using image dimensions
        h, w = outputs['image_size'] if 'image_size' in outputs else (224, 224)  # Add image size to outputs
        scale_factor = torch.tensor([w, h], device=src_points.device)
        src_points = src_points / scale_factor
        target_points = target_points / scale_factor

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_points'] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses

def build(args, training):
    num_classes = 1
    model = RiceSeedlingDetector(row=args["row"], line=args["line"])
    
    if not training:
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args["point_loss_coef"]}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)  # You'll need to implement this based on your matcher
    criterion = SetCriterion(num_classes=num_classes, 
                        matcher=matcher,
                        weight_dict=weight_dict,
                        eos_coef=args["eos_coef"],
                        losses=losses)

    return model, criterion